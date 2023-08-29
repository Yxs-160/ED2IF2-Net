# ED2IF2-Net
Code of ED2IF2-Net

The datasets need to be pre-processed for our training.
python ./preprocessing/process_normals.py
python ./preprocessing/estimate_density.py # pre-compute the point sampling density for our weighted sampling strategy during training.
python ./preprocessing/estimate_gradient.py # pre-compute the SDF gradients to classify the front-side and back-side nearby points.

For training and testing, please directly run
python train.py
and 
python test.py

To change the default settings, please see utils.py.

During training, it saves the reconstructions of some selected images with GT camera after every a few epochs, in "./ckpt/outputs/".

The test script is to reconstruct from the input image with inferred camera. Before testing, please make sure the trained models, including camera model and D2IM model, are in the './ckpt/models/'. We provide our pre-trained model [here](https://drive.google.com/drive/folders/1UMNDy_NA9bKqe6T_xcTnxRMiea4neWw-?usp=sharing).
