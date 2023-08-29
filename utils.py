import os
import cv2
import mcubes
import torch
import numpy as np
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='ED2IF2-Net')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--plot_every_batch', type=int, default=10)
    parser.add_argument('--save_every_epoch', type=int, default=20)
    parser.add_argument('--test_every_epoch', type=int, default=20)
    parser.add_argument('--load_pretrain', type=bool, default=True)

    parser.add_argument('--viewnum', type=int, default=36)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument('--mcube_znum', type=int, default=128)
    parser.add_argument('--test_pointnum', type=int, default=100000)

    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--cam_batch_size', type=int, default=16)
    parser.add_argument('--cam_lr', type=float, default=0.00005)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--sampling_mode', type=str, default='weighted')
    parser.add_argument('--exp_name', type=str, default='ed2if2')

    parser.add_argument('--data_dir', default='/home/ED2IF2-Net-datasets/DISN_split/')
    parser.add_argument('--h5_dir', default='/home/ED2IF2-Net-datasets/')
    parser.add_argument('--density_dir', default='/home/ED2IF2-Net-datasets/SDF_density/')
    parser.add_argument('--cam_dir', default='/home/ED2IF2-Net-datasets/image/')
    parser.add_argument('--image_dir', default='/home/ED2IF2-Net-datasets/image/')
    parser.add_argument('--normal_dir', default='/home/ED2IF2-Net-datasets/normal_processed/')

    parser.add_argument('--model_dir', default='./ckpt/models')
    parser.add_argument('--output_dir', default='./ckpt/outputs')
    parser.add_argument('--log', default='log.txt')

    # some selected chairs with details
    testlist = [
        {'cat_id': '03001627', 'shape_id': 'ed751e0c20f48b3226fc87e2982c8a2b', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': 'd72f27e4240bd7d0283b00891f680579', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '5fa533f71e7e041efebad4f49b26ec52', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '8bb332c5afe67f8d917b96045c9b6dea', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '9a82269e56737217e16571f1d370cad9', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': 'd9159a24fb0259b7febad4f49b26ec52', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': 'e642ac79a2517d0054f92a30b31f64e', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': 'caa330f85a6d6db7a17ae19fa77775ff', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '4171071eb30dcb412dd4967de4160123', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '1d9dbebcbb82682bf27a705edb2f9ba6', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': 'df51cb83f0e55b81d85934de5898f0b', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '7fe08cd7a9b76c1dcbde89e0c48a01bf', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '33c4f94e97c3fefd19fb4103277a6b93', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '764866604b035caacd0362ae35d1beb4', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '6bd633162adccd29c3bd24f986301745', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': 'ed7b1be61b8e78ac5d8eba92952b9366', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '4719c75b8ce30de84b3c42e318f3affc', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '6b32d3a9198f8b03d1dcc55e36186e4e', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': 'ed953e5d1ce7a4bee7697d561711bd2b', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': 'eb04d1bffae6038c4c7384dbb75cab0d', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '64f6991a3688f8a0e49fc3668cb02f74', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '5ee976518fc4f5c8664b3b9b23ddfcbc', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '72da95dd6a486a4d4056b9c3d62d1efd', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '18bf93e893e4069e4b3c42e318f3affc', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '92e6546c4aca4ed14b96b665a8ac321', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': 'c520bc9dde7c0a19d2afe8d5254a0d04', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '875925d42780159ffebad4f49b26ec52', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '953a6c4d742f1e44d1dcc55e36186e4e', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '64ef0e07129b6bc4c3bd24f986301745', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': 'b3a9c49a1924f99815f855bb1d7c4f07', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '73b7d6df845221fa9a2041f674671d05', 'cam_id': 32},
        {'cat_id': '03001627', 'shape_id': '6a00357a35d0574b8d7d306df70cbb46', 'cam_id': 32}
    ]

    args = parser.parse_args()
    args.testlist = testlist
    args.catlist = ['03001627', '02691156', '02828884', '02933112', '03211117', '03636649', '03691459', '04090263', '04256520', '04379243', '04530566','02958343', '04401088']
    return args


def print_log(log_fname, logline):
    f = open(log_fname, 'a')
    f.write(logline)
    f.write('\n')
    f.close()


def save_checkpoint(epoch, model, optimizer, bestloss, output_filename):
    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'bestloss': bestloss}
    torch.save(state, output_filename)


def load_checkpoint(cp_filename, model, optimizer=None):
    checkpoint = torch.load(cp_filename)
    model.load_state_dict(checkpoint['state_dict'])
    if (optimizer is not None):
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    if ('bestloss' in checkpoint.keys()):
        bestloss = checkpoint['bestloss']
    else:
        bestloss = 10000000
    return epoch, model, optimizer, bestloss


def load_model(cp_filename, model):
    checkpoint = torch.load(cp_filename)
    model.load_state_dict(checkpoint['state_dict'])
    epoch = checkpoint['epoch']
    return epoch, model


""" sample grid points in the 3D space [-0.5,0.5]^3 """


def sample_grid_points(xnum, ynum, znum):
    gridpoints = np.zeros((xnum, ynum, znum, 3))
    for i in range(xnum):
        for j in range(ynum):
            for k in range(znum):
                gridpoints[i, j, k, :] = [i, j, k]
    gridpoints[:, :, :, 0] = (gridpoints[:, :, :, 0] + 0.5) / xnum - 0.5
    gridpoints[:, :, :, 1] = (gridpoints[:, :, :, 1] + 0.5) / ynum - 0.5
    gridpoints[:, :, :, 2] = (gridpoints[:, :, :, 2] + 0.5) / znum - 0.5
    return gridpoints


""" render the occupancy field to 3 image views """


def render_grid_occupancy(fname, gridvalues, threshold=0):
    signmat = np.sign(gridvalues - threshold)
    img1 = np.clip((np.amax(signmat, axis=0) - np.amin(signmat, axis=0)) * 256, 0, 255).astype(np.uint8)
    img2 = np.clip((np.amax(signmat, axis=1) - np.amin(signmat, axis=1)) * 256, 0, 255).astype(np.uint8)
    img3 = np.clip((np.amax(signmat, axis=2) - np.amin(signmat, axis=2)) * 256, 0, 255).astype(np.uint8)

    fname_without_suffix = fname[:-4]
    cv2.imwrite(fname_without_suffix + '_1.png', img1)
    cv2.imwrite(fname_without_suffix + '_2.png', img2)
    cv2.imwrite(fname_without_suffix + '_3.png', img3)


""" marching cube """


def render_implicits(fname, gridvalues, threshold=0):
    vertices, triangles = mcubes.marching_cubes(-1.0 * gridvalues, threshold)
    vertices[:, 0] = ((vertices[:, 0] + 0.5) / gridvalues.shape[0] - 0.5)
    vertices[:, 1] = ((vertices[:, 1] + 0.5) / gridvalues.shape[1] - 0.5)
    vertices[:, 2] = ((vertices[:, 2] + 0.5) / gridvalues.shape[2] - 0.5)
    write_ply(fname, vertices, triangles)


def write_obj(fname, vertices, triangles):
    fout = open(fname, 'w')
    for ii in range(len(vertices)):
        fout.write("v " + str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + "\n")
    for ii in range(len(triangles)):
        fout.write("f " + str(int(triangles[ii, 0]) + 1) + " " + str(int(triangles[ii, 1]) + 1) + " " + str(
            int(triangles[ii, 2]) + 1) + "\n")
    fout.close()


def write_ply(fname, vertices, triangles):
    fout = open(fname, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face " + str(len(triangles)) + "\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + "\n")
    for ii in range(len(triangles)):
        fout.write("3 " + str(triangles[ii, 0]) + " " + str(triangles[ii, 1]) + " " + str(triangles[ii, 2]) + "\n")
    fout.close()
