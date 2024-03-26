Note: 
- All the results get saved in the output folder. (The pre-computed results can be found in the folder as well)
- The entirety of this readme file assumes that the 'src' folder is chosen as the root folder in the terminal.
- It is assumed that the required dependencies have been installed, and the right conda environment has been activated.
- The pre-trained checkpoints can be found at https://drive.google.com/drive/folders/1vbEQ_ABmnH0IhgkWFIo_Ee4t_IBp-j_f?usp=share_link. Please replace the empty ckpts folder with the downloaded one.

The following steps shall be followed to get the outputs as required:

1.1. Fitting a voxel grid:

- Run the following command in your terminal:

python fit_data.py --type 'vox'

- Output gets generated in outputs/fit_data folder

1.2. Fitting a point cloud:

- Run the following command in your terminal:

python fit_data.py --type 'point'

- Output gets generated in outputs/fit_data folder

1.1. Fitting a mesh:

- Run the following command in your terminal:

python fit_data.py --type 'mesh'

- Output gets generated in outputs/fit_data folder


2.1. Image to voxel grid:

- Run the following command in your terminal and add other parameters as required for training:

python train_model.py --type 'vox'

- Evaluate the model by running the folllowing command in the terminal

python eval_model.py --type 'vox' --load_checkpoint

- The results for 3 sets (GT image, GT mesh and predicted voxels) get generated in output/eval_model folder


2.2. Image to point cloud:

- Run the following command in your terminal and add other parameters as required for training:

python train_model.py --type 'point'

- Evaluate the model by running the folllowing command in the terminal

python eval_model.py --type 'point' --load_checkpoint

- The results for 3 sets (GT image, GT mesh and predicted point cloud) get generated in output/eval_model folder


2.3. Image to mesh:

- Run the following command in your terminal and add other parameters as required for training:

python train_model.py --type 'mesh'

- Evaluate the model by running the folllowing command in the terminal

python eval_model.py --type 'mesh' --load_checkpoint

- The results for 3 sets (GT image, GT mesh and predicted mesh) get generated in output/eval_model folder

2.4. Quantitative comparisons:

- Upon running the following command in the terminal, the F1 score plots for different 3D representations get generated in the plots/ folder

python eval_model.py --type voxel|mesh|point --load_checkpoint

2.5. Analysing effect of hyperparameter variations

- Run the following command with varying values of w_chamfer, w_smooth and n_points to train the model accordingly

python train_model.py --type 'mesh' --w_chamfer {your value} --w_smooth {your value} --n_points {your value}

- Run the following command to see your results in the /output folder

python eval_model.py --type 'mesh' --load_checkpoint --n_points {your value}

2.6. Model interpretation

- Run the following two commands in the terminal and see the results in output/train_vis folder

python train_model.py --type vox' --max_iter 120 --batch_size 32 --vis_flag --log_freq 1 --lr 4e-2

python train_model.py --type 'point' --max_iter 120 --n_points 5000 --batch_size 32 --vis_flag --lr 4e-5
