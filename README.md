# Single Image to 3D with 3D Ground Truth

Goals: In this project, we will explore the types of loss and decoder functions for regressing to voxels, point clouds, and mesh representation from single view RGB input. 

## Setup

Please download and extract the dataset from [here](https://drive.google.com/file/d/1VoSmRA9KIwaH56iluUuBEBwCbbq3x7Xt/view?usp=sharing).
After unzipping, set the appropiate path references in `dataset_location.py` file

```
conda create -n pytorch3d-env python=3.9
conda activate pytorch3d-env
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip install numpy PyMCubes matplotlib
```

Make sure you have installed the packages mentioned in `requirements.txt`.
This project will need the GPU version of pytorch.

## 1. Fitting using optimization
The following 3 sections help us fit a source 3D representation to the target 3D representation using a simple optimization pipeline (preliminary task without any deep learning component).

### 1.1. Fitting a voxel grid
For this, we have defined binary cross entropy loss that can help us <b>fit a 3D binary voxel grid</b> in 'losses.py' file.

Run the file `python fit_data.py --type 'vox'`, to fit the source voxel grid to the target voxel grid. 

### 1.2. Fitting a point cloud
For this we have defined chamfer loss that helps us <b> fit a 3D point cloud </b> in 'losses.py' file.

Run the file `python fit_data.py --type 'point'`, to fit the source point cloud to the target point cloud. 

### 1.3. Fitting a mesh
For this we define an additional smoothening loss that can help us <b> fit a mesh</b> in 'losses.py' file.

Run the file `python fit_data.py --type 'mesh'`, to fit the source mesh to the target mesh. 

## 2. Reconstructing 3D from single view
This section will involve training a single view to 3D pipeline for voxels, point clouds and meshes.
Refer to the `save_freq` argument in `train_model.py` to save the model checkpoint quicker/slower. 

The repo has pretrained ResNet18 features of images to save computation and GPU resources required. Use `--load_feat` argument to use these features during training and evaluation. This should be False by default, and one can use this if they are facing issues in getting GPU resources. You can also enable training on a CPU by the `device` argument.

### 2.1. Image to voxel grid
For this we define a neural network to decode binary voxel grids. The decoder network is defined in `model.py` file.

Run the file `python train_model.py --type 'vox'`, to train single view to voxel grid pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth voxel grid and predicted voxel in `eval_model.py` file using:
`python eval_model.py --type 'vox' --load_checkpoint`

### 2.2. Image to point cloud
For this we define a neural network to decode point clouds. The decoder network is defined in `model.py` file.

Run the file `python train_model.py --type 'point'`, to train single view to pointcloud pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth point cloud and predicted  point cloud in `eval_model.py` file using:
`python eval_model.py --type 'point' --load_checkpoint`

### 2.3. Image to mesh (20 points)
For this we define a neural network to decode meshes. The decoder network is defined in `model.py` file.

Run the file `python train_model.py --type 'mesh'`, to train single view to mesh pipeline, feel free to tune the hyperparameters as per your need.

After trained, visualize the input RGB, ground truth mesh and predicted mesh in `eval_model.py` file using:
`python eval_model.py --type 'mesh' --load_checkpoint`

### 2.4. Quantitative comparisions
One can quantitatively compare the F1 score of 3D reconstruction for meshes vs pointcloud vs voxelgrids by running the following evaluation script:

For evaluating you can run:
`python eval_model.py --type voxel|mesh|point --load_checkpoint`

### 2.5. Analyse effects of hyperparms variations
Effects of some hyperparameter variations can be found in the webpage.

### 2.6. Interpret your model
Simply seeing final predictions and numerical evaluations is not always insightful. Can you create some visualizations that help highlight what your learned model does? Be creative and think of what visualizations would help you gain insights. There is no `right' answer - although reading some papers to get inspiration might give you ideas.

