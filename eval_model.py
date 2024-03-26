import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
from utils import render_360, get_mesh_from_voxels, preprocess_mesh, preprocess_pcl
import matplotlib.pyplot as plt 
import cv2
# import plotly.graph_objects as go
import numpy as np

# python eval_model.py --type 'vox' --load_checkpoint
# python eval_model.py --type 'point' --load_checkpoint
# python eval_model.py --type 'mesh' --load_checkpoint


def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--vis_freq', default=3, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=str)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        feats = torch.stack(feed_dict['feats']).to(args.device)
        return images, feats, mesh
    
    else:
        return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'plots/eval_{args.type}', bbox_inches='tight')


def compute_sampling_metrics(pred_points, gt_points, thresholds, eps=1e-8):
    metrics = {}
    lengths_pred = torch.full(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = torch.full(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = torch.sigmoid(predictions)
        # print(voxels_src.mean())
        voxels_src = voxels_src.detach().cpu().squeeze().numpy()
        # voxels_src = voxels_src.flatten()
        # min = np.min(voxels_src)
        # max = np.max(voxels_src)
        # voxels_src = ((voxels_src - min)/(max - min)).reshape((32,32,32))

        H,W,D = voxels_src.shape
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src, isovalue=0.3)
        # print(vertices_src.shape)
        vertices_src = torch.tensor(vertices_src).float()
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points, H, W, D)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics



def evaluate_model(args):
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model =  SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    if args.load_checkpoint:
        checkpoint = torch.load(f"ckpts/{args.type}/checkpoint_{args.type}.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    n = 1
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        if args.load_feat:
            images_gt, feats_gt, mesh_gt = preprocess(feed_dict, args)
            predictions = model(feats_gt, args)

        else:
            images_gt, mesh_gt = preprocess(feed_dict, args)
            predictions = model(images_gt, args)

        read_time = time.time() - read_start_time

        if args.type == "vox":
            predictions = predictions.permute(0,1,4,3,2)

        metrics = evaluate(predictions, mesh_gt, thresholds, args)

        # TODO:
        if (step % args.vis_freq) == 0 and n < 4:
            # visualization block
            #  rend = 
            # plt.imsave(f'vis/{step}_{args.type}.png', rend)
            if args.type == 'vox':
                base = "outputs/eval_model/"
                mesh_src = get_mesh_from_voxels(predictions, normalize=True)
                mesh_gt = preprocess_mesh(mesh_gt)
                render_360(item = mesh_src, type3d = args.type, save_path = base + f"eval_model_{args.type}_pred_{n}.gif")
                render_360(item = mesh_gt, type3d = args.type, save_path = base + f"/eval_model_{args.type}_meshgt_{n}.gif")
                plt.imsave(base + f'eval_model_{args.type}_img_{n}.png', cv2.resize(images_gt.cpu().squeeze().numpy(), (512, 512)))
                n += 1

            if args.type == 'mesh':
                base = "outputs/eval_model/"
                mesh_src = preprocess_mesh(predictions)
                mesh_gt = preprocess_mesh(mesh_gt)
                render_360(item = mesh_src, type3d = args.type, save_path = base + f"eval_model_{args.type}_pred_{n}.gif")
                render_360(item = mesh_gt, type3d = args.type, save_path = base + f"/eval_model_{args.type}_meshgt_{n}.gif")
                plt.imsave(base + f'eval_model_{args.type}_img_{n}.png', cv2.resize(images_gt.cpu().squeeze().numpy(), (512, 512)))
                n += 1

            if args.type == 'point':
                base = "outputs/eval_model/"
                pcl_src = preprocess_pcl(predictions, normalize=True)
                mesh_gt = preprocess_mesh(mesh_gt)
                render_360(item = pcl_src, type3d = args.type, save_path = base + f"eval_model_{args.type}_pred_{n}.gif")
                render_360(item = mesh_gt, type3d = 'mesh', save_path = base + f"/eval_model_{args.type}_meshgt_{n}.gif")
                plt.imsave(base + f'eval_model_{args.type}_img_{n}.png', cv2.resize(images_gt.cpu().squeeze().numpy(), (512, 512)))
                n += 1

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        f1_05 = metrics['F1@0.050000']
        avg_f1_score_05.append(f1_05)
        avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    

    avg_f1_score = torch.stack(avg_f1_score).mean(0)

    save_plot(thresholds, avg_f1_score,  args)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
