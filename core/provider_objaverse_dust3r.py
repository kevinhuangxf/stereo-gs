import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from PIL import Image

import kiui
from kiui.cam import orbit_camera, OrbitCamera
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

from dust3r.utils.image import load_images

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

import objaverse
import pandas as pd
import open3d as o3d
import trimesh
import json
import glob
import math

from plyfile import PlyData
from pathlib import Path

objaverse.BASE_PATH = '/workspace/datasets/objaverse'
objaverse._VERSIONED_PATH = '/workspace/datasets/objaverse/hf-objaverse-v1'

# kiui_uids = pd.read_csv("/workspace/code/objaverse_filter/kiuisobj_v1_merged_80K.csv", header=None)
# uids = kiui_uids[1].values.tolist()

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def normalize_point_cloud(point_cloud: torch.Tensor, box_scale: float):
    # Assuming point_cloud is a tensor of shape (N, 3) where N is the number of points
    bbox_min, bbox_max = torch.min(point_cloud, dim=0)[0], torch.max(point_cloud, dim=0)[0]
    scale = box_scale / torch.max(bbox_max - bbox_min)
    # Scale the point cloud
    point_cloud *= scale
    # Recompute the bounding box
    bbox_min, bbox_max = torch.min(point_cloud, dim=0)[0], torch.max(point_cloud, dim=0)[0]
    offset = -(bbox_min + bbox_max) / 2
    # Translate the point cloud
    point_cloud += offset
    return point_cloud


class ObjaverseDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training

        # TODO: remove this barrier
        # self._warn()

        # TODO: load the list of objects for training
        # self.items = []
        # with open('TODO: file containing the list', 'r') as f:
        #     for line in f.readlines():
        #         self.items.append(line.strip())
        # self.items = uids

        # local_views_path_json= "/workspace/code/objaverse-rendering/valid_views.json"
        # local_views_path_json= "/workspace/code/objaverse-rendering/valid_views_miku.json"
        if self.training:
            local_views_path_json = self.opt.local_views_path_json
        else:
            local_views_path_json = self.opt.eval_views_path_json
        with open(local_views_path_json, 'r') as f:
            local_views = json.load(f)
        self.items = local_views
        if len(self.items) < 1000 and self.training:
            self.items = self.items * (1000 // len(self.items))

        local_models_path_json = self.opt.local_views_path_json
        if local_models_path_json is not None:
            with open(local_models_path_json, 'r') as f:
                local_models_path = json.load(f)
            uids = [item.split('/')[-1] for item in local_models_path]
            self.objects = objaverse.load_objects(uids=uids)
            # self.glb_list = local_models_path
            # if len(self.glb_list) < 1000:
            #     self.glb_list = self.glb_list * (1000 // len(self.glb_list))

        # naive split
        # if self.training:
        #     self.items = self.items[:-self.opt.batch_size]
        # else:
        #     self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1        

        # camera parameters
        self.camera_params = OrbitCamera(self.opt.input_size, self.opt.input_size, r=1.5, fovy=67.38)
        self.intrinsics = torch.from_numpy(self.camera_params.intrinsics) # np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    def __len__(self):
        return len(self.items)

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha

    def load_glb_o3d(self, glb_path, num_sample_points):
        # Load the GLB file using open3d
        mesh = o3d.io.read_triangle_mesh(glb_path)

        # Get the point positions
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=num_sample_points)

        # Get the point positions from the point cloud
        point_positions_tensor =  torch.tensor(pcd.points).float()
        point_positions_tensor[:, [1, 2]] = point_positions_tensor[:, [2, 1]]
        # point_positions_tensor = point_positions_tensor / (torch.abs(point_positions_tensor).max())
        point_positions_tensor = normalize_point_cloud(point_positions_tensor, 2.0)
        # self.glb_list.append(point_positions_tensor)
        return point_positions_tensor

    def load_glb(self, glb_path, num_sample_points=65536):
        # Load the .glb file
        scene_or_mesh = trimesh.load_mesh(glb_path, force='mesh')

        # Collect all PointCloud vertices
        point_clouds = [
            geometry for geometry in scene_or_mesh.geometry.values()
            if isinstance(geometry, trimesh.PointCloud)
        ]

        if len(point_clouds) > 0:
            print(f"Found {len(point_clouds)} PointCloud geometries.")
            combined_vertices = np.vstack([pc.vertices for pc in point_clouds])
            print(combined_vertices.shape)

            # Randomly sample points from the combined point cloud
            print(f"Sampling {num_sample_points} points from the combined point cloud.")
            sampled_indices = np.random.choice(len(combined_vertices), num_sample_points, replace=True)
            points = combined_vertices[sampled_indices]
            # points = points[np.newaxis, :, :]

            point_positions_tensor =  torch.tensor(points).float()
            point_positions_tensor = normalize_point_cloud(point_positions_tensor, 2.0)
            return point_positions_tensor


        # Check if the loaded object is a Scene
        if isinstance(scene_or_mesh, trimesh.Scene):
            # Combine all the meshes in the scene into a single mesh
            mesh = scene_or_mesh.dump(concatenate=True)
        else:
            # The loaded object is already a Mesh
            mesh = scene_or_mesh

        # Sample points from the mesh
        # points = mesh.sample(10000)
        
        if isinstance(mesh, trimesh.path.Path3D): 
            # Access the vertices
            vertices = mesh.vertices
            num_vertices = vertices.shape[0]
            # Randomly sample indices
            sampled_indices = np.random.choice(num_vertices, num_sample_points, replace=True)
            # Extract sampled points
            points = vertices[sampled_indices]
            points = points[np.newaxis, :, :]
        else:
            # If the mesh is a list, concatenate all the meshes into one
            points = trimesh.sample.sample_surface(mesh, num_sample_points)

        point_positions_tensor =  torch.tensor(points[0]).float()
        point_positions_tensor = normalize_point_cloud(point_positions_tensor, 2.0)
        return point_positions_tensor

    def load_depth(self, depth_path):
        dep_img = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if dep_img is None:
            print(f'[WARN] failed to load depth image {depth_path}')
        dep_img[dep_img > 100] = 0
        if dep_img.max() - dep_img.min() == 0:
            dep_img = np.zeros_like(dep_img)
        else:
            dep_img = (dep_img - dep_img.min())/(dep_img.max() - dep_img.min())
        dep_img = dep_img[:, :, 0:1]

        dep_img = torch.from_numpy(dep_img).permute(2, 0, 1).contiguous().float()
        return dep_img

    def __getitem__(self, idx):

        uid = self.items[idx]
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        depths = []

        vid_cnt = 0

        # load glb
        if self.opt.local_models_path_json is not None:
            if hasattr(self.opt, "num_sample_points"):
                num_sample_points = self.opt.num_sample_points
            else:
                num_sample_points = 32768
            point_cloud = self.load_glb(self.glb_list[idx], num_sample_points)

        # TODO: choose views, based on your rendering settings
        # if self.training:
        #     # input views are in (36, 72), other views are randomly selected
        #     vids = np.random.permutation(np.arange(36, 73))[:self.opt.num_input_views].tolist() + np.random.permutation(100).tolist()
        # else:
        #     # fixed views
        #     vids = np.arange(36, 73, 4).tolist() + np.arange(100).tolist()
        # vids = np.arange(0, 8, 1).tolist()
        if self.training:
            vids = np.random.permutation(32)[:self.opt.num_views].tolist() # self.opt.num_input_views
        else:
            vids = np.arange(self.opt.num_views).tolist()

        bkg_color = [1.0, 1.0, 1.0]

        image_path_list = []
        depth_path_list = []
        for vid in vids:

            image_path = os.path.join(uid, 'rgba', f'{vid:03d}.png')
            camera_path = os.path.join(uid, 'pose', f'{vid:03d}.npy')

            if self.opt.use_depth:
                depth_path = os.path.join(uid, 'depth', f'{vid:03d}_depth0001.exr')
                depth_path_list.append(depth_path)
                depth_img = self.load_depth(depth_path)
            try:
                # TODO: load data (modify self.client here)
                # image = np.frombuffer(self.client.get(image_path), np.uint8)
                # image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]
                image, alpha = self.load_im(image_path, bkg_color)
                # c2w = [float(t) for t in self.client.get(camera_path).decode().strip().split(' ')]
                # c2w = torch.tensor(c2w, dtype=torch.float32).reshape(4, 4)
                image_path_list.append(image_path)

                c2w = np.load(camera_path)
                c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]])], axis=0)

                # blender world + opencv cam --> opengl world & cam
                c2w[1] *= -1
                c2w[[1, 2]] = c2w[[2, 1]]

                c2w = torch.from_numpy(c2w).float()
            except Exception as e:
                print(f'[WARN] dataset {uid} {vid}: {e}')
                continue
            
            # c2w = torch.from_numpy(orbit_camera(0, 360 - vid * 90, radius=2.0, opengl=True))

            # TODO: you may have a different camera system
            # c2w = torch.linalg.inv(w2c)
            # blender world + opencv cam --> opengl world & cam
            # c2w[2] *= -1
            # c2w[[1, 2]] = c2w[[2, 1]]
            # c2w[:3, 1:3] *= -1 # invert up and forward direction

            # c2w_new = c2w.clone()
            # c2w_new[1] = c2w[2]
            # c2w_new[2] = c2w[1]
            # c2w = c2w_new

            # scale up radius to fully use the [-1, 1]^3 space!
            # c2w[:3, 3] *= self.opt.cam_radius / 1.5 # 1.5 is the default scale
        
            # disabled since we already get masks
            # image = image.permute(2, 0, 1) # [4, 512, 512]
            # mask = image[3:4] # [1, 512, 512]
            # image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            # image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(alpha.squeeze(0))
            cam_poses.append(c2w)
            if self.opt.use_depth:
                depths.append(depth_img)

            vid_cnt += 1
            if vid_cnt == self.opt.num_views:
                break

        if vid_cnt < self.opt.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.opt.num_views - vid_cnt
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]
        if self.opt.use_depth:
            depths = torch.stack(depths, dim=0) # [V, 4, 4]

        # camera augmentation
        
        if self.opt.camera_augmentation:
            # random rotation along z axis
            degree = np.random.uniform(0, 0.1 * math.pi * 2)
            rot = torch.tensor([
                [np.cos(degree), -np.sin(degree), 0, 0],
                [np.sin(degree), np.cos(degree), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]).unsqueeze(0).float()
            cam_poses = torch.matmul(rot, cam_poses)

            # random scaling
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.9, 1.1)
                cam_poses[:, :3, 3] *= scale

            # add noise to input cameras
            cam_poses[:, :3, :] = cam_poses[:, :3, :] + torch.rand_like(cam_poses[:, :3, :]) * 0.04 - 0.02

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        # transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[1])
        # cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        # cam_poses[0, 1, 1] = cam_poses[0, 1, 1] * -1

        # elevation = 0
        # cam_poses = []
        # for azi in [0, 90, 180, 270, 0, 90, 180, 270]:
        #     cam_pose = torch.from_numpy(orbit_camera(elevation, azi, radius=1.5, opengl=True)).unsqueeze(0)
        #     cam_pose[:, :3, 1:3] *= -1 # invert up & forward direction
        #     cam_poses.append(cam_pose[0])
        # cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # data augmentation
        if self.training or self.opt.data_mode == 's3':
            # apply random grid distortion to simulate 3D inconsistency
            # print("apply random grid distortion to simulate 3D inconsistency")
            if random.random() < self.opt.prob_grid_distortion or self.opt.data_mode == 's3':
                images_input[1:] = grid_distortion(images_input[1:])
                # print('distortion')
            # apply camera jittering (only to input!)
            # if random.random() < self.opt.prob_cam_jitter or self.opt.data_mode == 's3':
            #     cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])
            #     # print('jitter')

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        if self.opt.use_depth:
            results['depths_output'] = F.interpolate(depths, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        images = load_images(image_path_list[:self.opt.num_input_views], size=self.opt.output_size, square_ok=True, verbose=False)
        for i in range(self.opt.num_input_views):
            # images[i]['img'] = images[i]['img'].squeeze(0)
            # images[i]['true_shape'] = images[i]['true_shape'].squeeze(0)
            if i%2 == 0 and i != 0:
                images[0]['img'] = torch.cat((images[0]['img'], images[i]['img']), dim=0)
            elif i%2 == 1 and i != 1:
                images[1]['img'] = torch.cat((images[1]['img'], images[i]['img']), dim=0)

        results['images'] = images 
        H, W = images[0]['img'].shape[2:]

        # build rays for input views
        rays_embeddings = []
        first_views_id = [i for i in range(self.opt.num_input_views) if i % 2 == 0]
        second_views_id = [i for i in range(self.opt.num_input_views) if i % 2 != 0]
        # for i in range(self.opt.num_input_views):
            # rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
        for i in first_views_id + second_views_id:
            rays_o, rays_d = get_rays(cam_poses_input[i], H, W, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        # rays_embeddings[[1, 2], :, :, :] = rays_embeddings[[2, 1], :, :, :]
        results['rays'] = rays_embeddings
        if self.opt.data_mode == 's3':
            rays_embeddings = []
            for i in range(self.opt.num_input_views):
                rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
                rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
                rays_embeddings.append(rays_plucker)
        
            rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
            final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
            results['input'] = final_input
        # results['input'] = images_input

        # depth_gt for input views
        if self.opt.use_depth:
            depth_imgs_inp = []
            for i in first_views_id + second_views_id:
                dep_img = self.load_depth(depth_path_list[i])
                depth_imgs_inp.append(dep_img)
            depth_imgs_inp = torch.stack(depth_imgs_inp, dim=0) # [V, 6, h, w]
            results['depths_input'] = F.interpolate(depth_imgs_inp, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]

        # mask_gt for input views
        masks_inp = []
        for i in first_views_id + second_views_id:
            masks_inp.append(masks[i])
        masks_inp = torch.stack(masks_inp, dim=0) # [V, h, w]
        results['masks_input'] = F.interpolate(masks_inp.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        intrinsics = torch.zeros(3, 3)
        intrinsics[0, 0] = self.intrinsics[0]
        intrinsics[1, 1] = self.intrinsics[1]
        intrinsics[0, 2] = self.intrinsics[2]
        intrinsics[2, 2] = self.intrinsics[3]
        results['intrinsics'] = intrinsics.unsqueeze(0).repeat(self.opt.num_views, 1, 1)
        results['extrinsics'] = cam_poses

        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos
        
        if self.opt.local_models_path_json is not None:
            results['point_cloud'] = point_cloud

            # glb_path = "/workspace/code/objaverse-rendering/results/hatsune_miku/hatsune_miku.glb"
            # point_cloud = self.load_glb(glb_path, num_sample_points=10000)
            # results['point_cloud'] = point_cloud

            results['point_cloud'] = self.load_glb(self.objects[uid.split('/')[-1]], num_sample_points=10000)

        return results

class ShapeSplatDataset(Dataset):

    def __init__(self, opt: Options, training=True):
        
        self.opt = opt
        self.training = training

        # load ply path
        ply_folder = Path(self.opt.ply_files_path)
        self.items = [p.as_posix() for p in ply_folder.glob('*.ply')]

        # TODO: load the list of objects for training
        # self.items = []
        # with open('TODO: file containing the list', 'r') as f:
        #     for line in f.readlines():
        #         self.items.append(line.strip())
        # self.items = uids


        # local_views_path_json = self.opt.local_views_path_json
        # with open(local_views_path_json, 'r') as f:
        #     local_views = json.load(f)
        # self.items = local_views
        # if len(self.items) < 1000:
        #     self.items = self.items * (1000 // len(self.items))

        # naive split
        # if self.training:
        #     self.items = self.items[:-self.opt.batch_size]
        # else:
        #     self.items = self.items[-self.opt.batch_size:]
        
        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1        


    def __len__(self):
        return len(self.items)

    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha

    def load_ply(self, ply_path):
        gs_vertex = PlyData.read(ply_path)['vertex']

        ### load centroids[x,y,z] - Gaussian centroid
        x = gs_vertex['x'].astype(np.float32)
        y = gs_vertex['y'].astype(np.float32)
        z = gs_vertex['z'].astype(np.float32)
        centroids = np.stack((x, y, z), axis=-1) # [n, 3]

        ### load o - opacity
        opacity = gs_vertex['opacity'].astype(np.float32).reshape(-1, 1)

        ### load scales[sx, sy, sz] - Scale
        scale_names = [
            p.name
            for p in gs_vertex.properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((centroids.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = gs_vertex[attr_name].astype(np.float32)

        ### load rotation rots[q_0, q_1, q_2, q_3] - Rotation
        rot_names = [
            p.name for p in gs_vertex.properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((centroids.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = gs_vertex[attr_name].astype(np.float32)

        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)

        ### load base sh_base[dc_0, dc_1, dc_2] - Spherical harmonic
        sh_base = np.zeros((centroids.shape[0], 3, 1))
        sh_base[:, 0, 0] = gs_vertex['f_dc_0'].astype(np.float32)
        sh_base[:, 1, 0] = gs_vertex['f_dc_1'].astype(np.float32)
        sh_base[:, 2, 0] = gs_vertex['f_dc_2'].astype(np.float32)
        sh_base = sh_base.reshape(-1, 3)

        centroids, opacity, scales, rots, sh_base = torch.from_numpy(centroids), torch.from_numpy(opacity), torch.from_numpy(scales), torch.from_numpy(rots), torch.from_numpy(sh_base)
        gaussians = torch.cat([centroids, opacity, scales, rots, sh_base], dim=-1) # [B, N, 14]

        return gaussians

    def __getitem__(self, idx):

        results = {}

        # load ply
        ply_path = self.items[idx]
        gaussians = self.load_ply(ply_path)

        results['gaussians'] = gaussians
        
        # generate camera poses
        cam_poses = []
        elevations = [random.randint(-45, 45) for i in range(self.opt.num_input_views)]
        azimuths = [random.randint(-180, 180) for i in range(self.opt.num_input_views)]
        for ele, azi in zip(elevations, azimuths):
            cam_pose = torch.from_numpy(orbit_camera(ele, azi, radius=1.5, opengl=True)).unsqueeze(0)
            cam_poses.append(cam_pose[0])
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # # data augmentation
        # if self.training:
        #     # apply random grid distortion to simulate 3D inconsistency
        #     if random.random() < self.opt.prob_grid_distortion:
        #         images_input[1:] = grid_distortion(images_input[1:])
        #     # apply camera jittering (only to input!)
        #     if random.random() < self.opt.prob_cam_jitter:
        #         cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        # build rays for input views
        rays_embeddings = []
        first_views_id = [i for i in range(self.opt.num_input_views) if i % 2 == 0]
        second_views_id = [i for i in range(self.opt.num_input_views) if i % 2 != 0]
        # for i in range(self.opt.num_input_views):
            # rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
        for i in first_views_id + second_views_id:
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        # rays_embeddings[[1, 2], :, :, :] = rays_embeddings[[2, 1], :, :, :]
        results['rays'] = rays_embeddings

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        return results

