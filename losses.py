import torch
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss import mesh_laplacian_smoothing, chamfer_distance

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	loss_fn = torch.nn.BCEWithLogitsLoss()

	loss = loss_fn(voxel_src, voxel_tgt)

	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	knn_src = knn_points(point_cloud_src, point_cloud_tgt)
	knn_tgt = knn_points(point_cloud_tgt, point_cloud_src)

	loss_chamfer = knn_src.dists[...,0].mean() + knn_tgt.dists[...,0].mean()
	# loss_chamfer = chamfer_distance(point_cloud_src, point_cloud_tgt)[0]
	# print(loss_chamfer[0])

	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss
	loss_laplacian = mesh_laplacian_smoothing(mesh_src, method="uniform")

	return loss_laplacian