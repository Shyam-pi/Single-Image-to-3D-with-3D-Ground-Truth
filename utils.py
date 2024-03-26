import torch
import pytorch3d
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from pytorch3d.io import load_obj
import mcubes
import imageio
from matplotlib import pyplot as plt

def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def get_points_renderer(
    image_size=512, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

def render_360(item, type3d, save_path, camera_distance = 2.0, elevation = 10.0, num_frames = 100):
    azimuths = torch.linspace(0,360,num_frames)

    if type3d == "point": 
        renderer = get_points_renderer()
    else: 
        renderer = get_mesh_renderer()

    item = item.to(device = "cuda")
    lights = pytorch3d.renderer.PointLights(location=[[0, 4.0, 0.0]], device="cuda")
    images = []

    # Loop through different camera angles, render each frame, and save them as images
    for i, azimuth in enumerate(azimuths):
        # print(i)
        # Update the camera pose for the current frame
        R, T = pytorch3d.renderer.look_at_view_transform(camera_distance, elevation, azimuth, device="cuda")
        cameras = pytorch3d.renderer.PerspectiveCameras(R=R, T=T, device="cuda")

        # Render the frame
        rend = renderer(item, cameras=cameras, lights = lights)
        rend = (rend.cpu().numpy()[0, ..., :3] * 255).astype("uint8")

        images.append(rend)

    imageio.mimsave(save_path, images, duration=0.01, loop = 0)

def render_rgb(item, type3d, save_path):

    if type3d == "point": 
        renderer = get_points_renderer()
    else: 
        renderer = get_mesh_renderer()

    item = item
    lights = pytorch3d.renderer.PointLights(location=[[0, -4.0, 0.0]], device="cuda")

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=80, device='cuda'
        )
    rend = renderer(item, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    output_path = save_path
    plt.imsave(output_path, rend)


def get_mesh_from_voxels(voxels, normalize = False):

    if normalize:
        voxels = torch.sigmoid(voxels)
        # print(voxels.mean())

    voxels = voxels.detach().cpu().squeeze().numpy()
    # print(voxels.flatten().max())
    # print(voxels.shape)
    mean = voxels.flatten().mean()
    vertices, faces = mcubes.marching_cubes(voxels, isovalue=mean)
    # print(vertices.shape)
    vertices = torch.tensor(vertices).float()
    vertices = vertices - vertices.mean(0)

    max_distance = torch.max(torch.norm(vertices, dim=1))
    vertices = vertices / max_distance

    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    # vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    color = [0.7, 0.7, 1]
    textures = torch.ones_like(vertices).unsqueeze(0)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    textures = pytorch3d.renderer.TexturesVertex(textures)

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device = "cuda"
    )

    return mesh

def preprocess_mesh(mesh):
    # print(vars(mesh))
    vertices = mesh._verts_list[0].detach()
    faces = mesh._faces_list[0].to(device = 'cuda')
    vertices = (vertices - vertices.mean(0))
    max_distance = torch.max(torch.norm(vertices, dim=1))
    vertices = (vertices / max_distance).to(device = 'cuda')

    color = [0.7, 0.7, 1]
    textures = torch.ones_like(vertices).unsqueeze(0)  # (1, N_v, 3)
    textures = textures * torch.tensor(color).to(device = 'cuda')  # (1, N_v, 3)
    textures = pytorch3d.renderer.TexturesVertex(textures).to(device = 'cuda')

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device = "cuda"
    )

    return mesh

def preprocess_pcl(points, normalize = True):
    """
    Function to center points around (0,0,0) and normalize distance of points from origin between 0 and 1
    """
    points = points.detach().squeeze(0)
    
    if normalize:
        points = points - points.mean(0)
        max_distance = torch.max(torch.norm(points, dim=1))
        points = (points / max_distance).to(device = 'cuda')

    else:
        points = points.to(device = 'cuda')

    color = [0.7, 0.7, 1]
    textures = torch.ones_like(points).to(device = 'cuda')  # (N_v, 3)
    textures = textures * torch.tensor(color).to(device = 'cuda')  # (N_v, 3)

    pcl = pytorch3d.structures.Pointclouds(points=points.unsqueeze(0), features=textures.unsqueeze(0))

    return pcl