from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import torch.nn.functional as F
import pytorch3d
import numpy as np

# class VoxDecoder(torch.nn.Module):
#     def __init__(self):
#         super(VoxDecoder, self).__init__()

#          # Layer Definition
#         self.layer1 = torch.nn.Sequential(
#             torch.nn.ConvTranspose3d(512, 128, kernel_size=2, stride=2, bias=True),
#             torch.nn.BatchNorm3d(128),
#             torch.nn.ReLU()
#         )
#         self.layer2 = torch.nn.Sequential(
#             torch.nn.ConvTranspose3d(128, 32, kernel_size=2, stride=2, bias=True),
#             torch.nn.BatchNorm3d(32),
#             torch.nn.ReLU()
#         )
#         self.layer3 = torch.nn.Sequential(
#             torch.nn.ConvTranspose3d(32, 8, kernel_size=2, stride=2, bias=True),
#             torch.nn.BatchNorm3d(8),
#             torch.nn.ReLU()
#         )
#         self.layer4 = torch.nn.Sequential(
#             torch.nn.ConvTranspose3d(8, 1, kernel_size=4, stride=2, bias=True, padding=1),  # Adjust kernel size and stride
#             torch.nn.BatchNorm3d(1),
#             torch.nn.Sigmoid()
#         )

#     def forward(self, input_features):
#         # Reshape the input to match the expected shape (B, 512, 1, 1, 1)
#         input_features = input_features.view(-1, 512, 1, 1, 1)

#         gen_volume = self.layer1(input_features)
#         print(gen_volume.shape)
#         gen_volume = self.layer2(gen_volume)
#         print(gen_volume.shape)
#         gen_volume = self.layer3(gen_volume)
#         print(gen_volume.shape)
#         gen_volume = self.layer4(gen_volume)
#         print(gen_volume.shape)

#         return gen_volume

class VoxDecoder(torch.nn.Module):
    def __init__(self):
        super(VoxDecoder, self).__init__()

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(512, 512*4*4*4, bias=True),  # Set bias to True
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True),  # Set bias to True
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True),  # Set bias to True
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True),  # Set bias to True
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=True),
            torch.nn.BatchNorm3d(1)
            # torch.nn.ReLU()
        )

    def forward(self, input_vectors):
        # Apply fully connected layer
        features = self.layer1(input_vectors)
        # Reshape the output for 2x2x2 upscaling
        features = features.view(-1, 512, 4, 4, 4)

        # Apply transposed convolutional layers to upscale to 32x32x32
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        voxels = self.layer5(features)

        return voxels

class PointCloudDecoder(nn.Module):
    def __init__(self, dim, n_points, dropout_prob=0.5):
        super(PointCloudDecoder, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            # nn.BatchNorm1d(512),  # Batch Normalization
            nn.Linear(512, 1024),
            nn.ReLU(),
            # nn.BatchNorm1d(1024),  # Batch Normalization
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_points * 3)  # 3 dimensions for each point
        )

    #     # Initialize weights
    #     self.init_weights()

    # def init_weights(self):
    #     # Custom weight initialization for better training stability
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #             nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(x).view(x.size(0), -1, 3)
    
# class PointCloudDecoder(nn.Module):
#     def __init__(self, dim, n_points, dropout_prob=0.5):
#         super(PointCloudDecoder, self).__init__()
        
#         self.fc = nn.Sequential(
#             nn.Linear(dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, 2048),
#             nn.ReLU(),
#             nn.Linear(2048, n_points * 3)  # 3 dimensions for each point
#         )

#         # Initialize weights
#         self.init_weights()

#     def init_weights(self):
#         # Custom weight initialization for better training stability
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 nn.init.zeros_(m.bias)

#     def forward(self, x):
#         return self.fc(x).view(x.size(0), -1, 3)


class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            pass
            # TODO:
            self.decoder = VoxDecoder()
            # self.decoder = VoxDecoder(dim=512)

        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  
            self.n_point = args.n_points
            # TODO:
            # self.decoder =             
            ###################
            # An example decoder:
            # mlp_0: 512 -> 512
            # ReLU
            # mlp_1: 512 -> 1024
            # ReLU
            # mlp_2: 1024 -> 2048
            # ReLU
            # mlp_3: 2048 -> 2048
            # ReLU
            # mlp_4: 2048 -> 2048
            # ReLU
            # mlp_5: 2048 -> N*3
            ###################
            self.decoder = PointCloudDecoder(dim=512, n_points=self.n_point)
            # self.decoder = ImplicitMLPDecoder(args, hidden_size=256, out_dim=3)
            
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            self.decoder = PointCloudDecoder(dim=512, n_points=mesh_pred.verts_packed().shape[0])
            # TODO:
            # self.decoder =   
            ###################
            # An example decoder:
            # mlp_0: 512 -> 512
            # ReLU
            # mlp_1: 512 -> 1024
            # ReLU
            # mlp_2: 1024 -> 2048
            # ReLU
            # mlp_3: 2048 -> 2048
            # ReLU
            # mlp_4: 2048 -> 2048
            # ReLU
            # mlp_5: 2048 -> N*3
            ###################          

    def forward(self, images, args):

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            voxels_pred = self.decoder(encoded_feat)            
            return voxels_pred

        elif args.type == "point":
            # TODO:
            pointclouds_pred = self.decoder(encoded_feat)            
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)           
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          