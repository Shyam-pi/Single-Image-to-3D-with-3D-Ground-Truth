import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
from pytorch3d.ops import sample_points_from_meshes
import losses
from pytorch3d.loss import chamfer_distance
from utils import get_mesh_from_voxels, render_360, preprocess_pcl

# python train_model.py --type 'vox' --max_iter 15000 --lr 4e-6 --batch_size 1 --load_checkpoint
# python train_model.py --type 'mesh' --max_iter 10000
# python train_model.py --type 'mesh' --max_iter 10000 --save_freq 100 --w_chamfer 5.0 --w_smooth 5.0
# python train_model.py --type 'point' --max_iter 10000

# python train_model.py --type 'point' --max_iter 120 --n_points 5000 --batch_size 32 --vis_flag

# python train_model.py --type vox' --max_iter 120 --batch_size 32 --vis_flag --log_freq 1 --lr 4e2

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    # Model parameters
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--log_freq', default=50, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=str)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)
    parser.add_argument('--save_freq', default=100, type=int)    
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument('--load_checkpoint', action='store_true')     
    parser.add_argument('--vis_flag', action='store_true')       
    return parser

def preprocess(feed_dict,args):
    images = feed_dict['images'].squeeze(1)
    if args.type == "vox":
        voxels = feed_dict['voxels'].float()
        ground_truth_3d = voxels
    elif args.type == "point":
        mesh = feed_dict['mesh']
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)    
        ground_truth_3d = pointclouds_tgt        
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]
    if args.load_feat:
        feats = torch.stack(feed_dict['feats'])
        return feats.to(args.device), ground_truth_3d.to(args.device)
    else:
        return images.to(args.device), ground_truth_3d.to(args.device)




def calculate_loss(predictions, ground_truth, args):
    if args.type == 'vox':
        loss = losses.voxel_loss(predictions,ground_truth)
    elif args.type == 'point':
        loss = args.w_chamfer * losses.chamfer_loss(predictions, ground_truth)
    elif args.type == 'mesh':
        sample_trg = sample_points_from_meshes(ground_truth, args.n_points)
        sample_pred = sample_points_from_meshes(predictions, args.n_points)

        loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
        loss_smooth = losses.smoothness_loss(predictions)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth        
    return loss


def train_model(args):
    r2n2_dataset = R2N2("train", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    train_loader = iter(loader)

    model =  SingleViewto3D(args)
    model.to(args.device)
    model.train()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)  # to use with ViTs
    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        checkpoint = torch.load(f"ckpts/{args.type}/checkpoint_{args.type}.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint['step']
        print(f"Succesfully loaded iter {start_iter}")

    print(len(train_loader))
    
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        if step % len(train_loader) == 0: #restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        feed_dict = next(train_loader)

        images_gt, ground_truth_3d = preprocess(feed_dict,args)
        
        if args.vis_flag and step == 0:
            test_in = images_gt.detach().clone()[0].unsqueeze(0)
            print(test_in.shape)

        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, args)

        if args.vis_flag and step % 10 == 0:
            if args.type == 'vox':
                base = "outputs/train_vis/train_vis_batch32_lr4e-2_vox/"
                pred = model(test_in, args)
                mesh_src = get_mesh_from_voxels(pred, normalize=True)
                render_360(item = mesh_src, type3d = args.type, save_path = base + f"train_vis_{args.type}_step_{step}.gif")

            if args.type == 'point':
                base = "outputs/train_vis/train_vis_batch32_lr4e-2_point/"
                pred = model(test_in, args)
                pcl_src = preprocess_pcl(pred, normalize=True)
                render_360(item = pcl_src, type3d = args.type, save_path = base + f"train_vis_{args.type}_step_{step}.gif")

        # print(f"prediction shape = {prediction_3d.shape}, gt shape = {ground_truth_3d.shape}")

        loss = calculate_loss(prediction_3d, ground_truth_3d, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        if (step % args.save_freq) == 0 and step != 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, f'ckpts/{args.type}/checkpoint_{args.type}.pth')

        if (step % args.log_freq) == 0: 
            print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f" % (step, args.max_iter, total_time, read_time, iter_time, loss_vis))

    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
