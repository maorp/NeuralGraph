import sys, os

import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import shutil
import copy
import time
import glob
from skimage import io
import json, codecs
from timeit import default_timer as timer
import math
from tqdm import tqdm
import argparse
import math

import marching_cubes as mcubes
from dataset.dataset import MeshDataset
import config as cfg
from utils.pcd_utils import (BBox,
                             transform_pointcloud_to_opengl_coords,
                             load_and_transform_mesh_to_unit_cube,
                             rotate_around_axis,
                             origin, normalize_transformation)
from utils.viz_utils import (visualize_grid, visualize_mesh, Colors, merge_line_sets, merge_meshes)
from utils.sdf_utils import (sample_grid_points, scale_grid)
from utils.line_mesh import LineMesh
from nnutils.node_proc import convert_embedding_to_explicit_params, compute_inverse_occupancy, sample_rbf_surface
from utils.parser_utils import check_non_negative, check_positive
from nnutils.geometry import augment_grid

import open3d as o3d

from node_sampler.model import NodeSampler
from multi_sdf.model import MultiSDF
from node_sampler.loss import SurfaceConsistencyLoss


class Viewer:

    def __init__(self, checkpoint_path, time_inc=1, gt_data_dir=None, \
                 grid_dim=128, grid_num_chunks=256, num_neighbors=1, edge_weight_threshold=0.0, viz_only_graph=False):
        self.time = 0
        self.time_inc = time_inc
        self.obj_mesh = None
        self.gt_mesh = None
        self.sphere_mesh = None
        self.edge_mesh = None
        self.show_gt = False
        self.show_spheres = False
        self.grid_dim = grid_dim
        self.grid_num_chunks = grid_num_chunks
        self.num_neighbors = num_neighbors
        self.viz_edges = num_neighbors > 0
        self.edge_weight_threshold = edge_weight_threshold
        self.viz_only_graph = viz_only_graph

        self.initialize(checkpoint_path, gt_data_dir)

    def initialize(self, checkpoint_path, gt_data_dir):
        ###############################################################################################
        # Paths.
        ###############################################################################################
        gt_data_paths = [os.path.join(gt_data_dir, gt_mesh) for gt_mesh in sorted(os.listdir(gt_data_dir)) if
                         os.path.isdir(os.path.join(gt_data_dir, gt_mesh))]

        num_gt_meshes = len(gt_data_paths)
        num_time_steps = math.ceil(float(num_gt_meshes) / self.time_inc)

        if num_time_steps > 1:
            time_steps = np.linspace(-1.0, 1.0, num_time_steps).tolist()
        else:
            time_steps = [0.0]

        time_idxs = []
        for t in range(len(time_steps)):
            time_idxs.append(t * self.time_inc)

        print("Time steps:", time_steps)
        print("Time idxs:", time_idxs)

        ###############################################################################################
        # Load model.
        ###############################################################################################
        assert os.path.isfile(
            checkpoint_path), "\nModel {} does not exist. Please train a model from scratch or specify a valid path to a model.".format(
            model_path)
        pretrained_dict = torch.load(checkpoint_path)

        # Check if the provided checkpoint is graph-only model, or complete multi-sdf model.
        only_graph_model = True
        for k in pretrained_dict:
            if "node_sampler" in k:
                only_graph_model = False
                break

        if only_graph_model:
            model = NodeSampler().cuda()
        else:
            model = MultiSDF().cuda()

        # Initialize weights from checkpoint.
        model.load_state_dict(pretrained_dict)

        # Put model into evaluation mode.
        model.eval()

        # Count parameters.
        n_all_model_params = int(sum([np.prod(p.size()) for p in model.parameters()]))
        n_trainable_model_params = int(
            sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
        print("Number of parameters: {0} / {1}".format(n_trainable_model_params, n_all_model_params))
        print()

        if only_graph_model:
            node_sampler = model
        else:
            node_sampler = model.node_sampler

        ###############################################################################################
        # Load groundtruth data (meshes and bounding boxes).
        ###############################################################################################
        print("Loading groundtruth data ...")

        self.gt_meshes = []
        self.grid_array = []
        self.rotated2gaps_array = []
        self.world2grid_array = []

        for time_idx in tqdm(time_idxs):
            gt_data_path = gt_data_paths[time_idx]

            # Load groundtruth mesh.
            orig2world = np.reshape(np.loadtxt(f'{gt_data_path}/orig_to_gaps.txt'), [4, 4])
            world2orig = np.linalg.inv(orig2world)

            gt_mesh = o3d.io.read_triangle_mesh(f'{gt_data_path}/mesh_orig.ply')
            gt_mesh.transform(orig2world)
            gt_mesh.compute_vertex_normals()
            gt_mesh.paint_uniform_color([0.5, 0.0, 0.0])

            # Transform to OpenGL coordinates
            # gt_mesh = rotate_around_axis(gt_mesh, axis_name="x", angle=-np.pi)
            self.gt_meshes.append(gt_mesh)

            # Load data from directory.
            grid, world2grid = MeshDataset.load_grid(f'{gt_data_path}/coarse_grid.grd')

            # Store loaded data.
            self.rotated2gaps_array.append(np.eye(4).astype(np.float32))
            self.world2grid_array.append(world2grid)
            self.grid_array.append(grid)

        ###############################################################################################
        # Evaluate node sampler.
        ###############################################################################################
        embeddings = []
        affinity_matrix_array = []
        rots = []
        cents = []
        conts = []
        scals = []

        print("Predicting embedding vectors and descriptors ...")

        for i in tqdm(range(len(self.grid_array))):
            t = time_idxs[i]
            with torch.no_grad():
                # Move to device
                grid = torch.from_numpy(self.grid_array[i]).cuda().unsqueeze(0)
                rotated2gaps = torch.from_numpy(self.rotated2gaps_array[i]).cuda().unsqueeze(0)
                world2grid = torch.from_numpy(self.world2grid_array[i]).cuda().unsqueeze(0)

                # Compute augmented sdfs.
                sdfs = augment_grid(grid, world2grid, rotated2gaps)

                # Forward pass.
                embedding_pred, source_idxs, target_idxs, pair_distances, pair_weights, affinity_matrix = node_sampler(
                    sdfs)
                embeddings.append(embedding_pred)

                affinity_matrix_union = torch.sum(affinity_matrix, dim=0) / float(cfg.num_neighbors)
                affinity_matrix_array.append(affinity_matrix_union)

                # Compute explicit parameters.
                constants, scales, rotations, centers = convert_embedding_to_explicit_params(embeddings[i],
                                                                                             rotated2gaps,
                                                                                             cfg.num_nodes,
                                                                                             cfg.scaling_type)
                rots.append(rotations)
                scals.append(scales)
                cents.append(centers)
                conts.append(constants)

                # Transform centers to grid cs.
                centers = centers.view(cfg.num_nodes, 3, 1)
                A_world2grid = world2grid[:, :3, :3].view(1, 3, 3).expand(cfg.num_nodes, -1, -1)
                t_world2grid = world2grid[:, :3, 3].view(1, 3, 1).expand(cfg.num_nodes, -1, -1)

                ########################################################################################

        catRot = torch.cat(rots, 0)
        catConsts = torch.cat(conts, 0)
        catScales = torch.cat(scals, 0)
        catCenters = torch.cat(cents, 0)

        print("transform source frame and write meshes...")
        srcFrame = 6
        points = np.asarray(self.gt_meshes[srcFrame].vertices)
        points = points.astype(np.float32)

        num_points = points.shape[0]
        rotated2gaps = torch.from_numpy(self.rotated2gaps_array[srcFrame]).cuda().unsqueeze(0)

        for trgFrame in range(len(self.grid_array)):

            trans_cloud = SurfaceConsistencyLoss.transform_cloud(catConsts, catScales, catRot, catCenters, points,
                                                                 srcFrame, trgFrame)
            trans_cloud_np = trans_cloud.cpu().numpy()
            mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(trans_cloud_np),
                                             self.gt_meshes[srcFrame].triangles)
            mesh.compute_vertex_normals()
            o3d.io.write_triangle_mesh(f'{gt_data_dir}/mesh_trans{trgFrame}.ply', mesh)
            o3d.io.write_triangle_mesh(f'{gt_data_dir}/mesh_trans{trgFrame}_gt.ply', self.gt_meshes[trgFrame])

        # srcFrame = 15
        # trgFrame = 3
        # points = np.asarray(self.gt_meshes[srcFrame].vertices)
        # points = points.astype(np.float32)
        #
        # num_points = points.shape[0]
        # rotated2gaps = torch.from_numpy(self.rotated2gaps_array[srcFrame]).cuda().unsqueeze(0)
        #
        # trans_cloud = SurfaceConsistencyLoss.transform_cloud(catConsts, catScales, catRot, catCenters, points, srcFrame, trgFrame)
        # trans_cloud_np = trans_cloud.cpu().numpy()
        #
        # mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(trans_cloud_np), self.gt_meshes[srcFrame].triangles)
        # mesh.compute_vertex_normals()
        #
        # o3d.visualization.draw_geometries([self.gt_meshes[srcFrame]])
        # o3d.visualization.draw_geometries([mesh])
        # o3d.visualization.draw_geometries([self.gt_meshes[trgFrame]])
        #
        # o3d.io.write_triangle_mesh(f'{gt_data_dir}/mesh_trans{trgFrame}.ply', mesh)
        # o3d.io.write_triangle_mesh(f'{gt_data_dir}/mesh_trans{srcFrame}.ply', self.gt_meshes[srcFrame])
        # o3d.io.write_triangle_mesh(f'{gt_data_dir}/mesh_trans{trgFrame}_gt.ply', self.gt_meshes[trgFrame])

        ###############################################################################################
        # Generate node spheres.
        ###############################################################################################
        print("Generating node spheres ...")

        constant_threshold = 0.0  # -0.07

        const_colors = np.asarray([
            [3.50792014e-01, 6.09477877e-01, 9.18623692e-02]
        ])

        self.node_meshes = []
        self.edge_meshes = []
        for i in tqdm(range(len(embeddings))):
            # Compute explicit parameters.
            rotated2gaps_i = torch.from_numpy(self.rotated2gaps_array[i]).cuda().unsqueeze(0)
            constants, scales, rotations, centers = convert_embedding_to_explicit_params(embeddings[i], rotated2gaps_i,
                                                                                         cfg.num_nodes,
                                                                                         cfg.scaling_type)

            if affinity_matrix_array[i] is not None:
                affinity_matrix = affinity_matrix_array[i].cpu().detach().numpy()
                self.viz_edges = True
            else:
                self.viz_edges = False

            # Generate sphere meshes.
            sphere_meshes = []
            for node_id in range(cfg.num_nodes):
                constant = constants[0, node_id].cpu().numpy()
                center = centers[0, node_id].cpu().numpy()
                rotation = rotations[0, node_id].cpu().numpy()
                scale = scales[0, node_id].cpu().numpy()

                if constant > constant_threshold:
                    continue

                # Create a sphere mesh.
                mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)

                scale = scale + 1e-8

                T_t = np.eye(4)
                T_t[0:3, 3] = center

                T_s = np.eye(4)
                T_s[0, 0] = scale[0]
                T_s[1, 1] = scale[1]
                T_s[2, 2] = scale[2]

                T_R = np.eye(4)

                T = np.matmul(T_t, np.matmul(T_R, T_s))
                mesh_sphere.transform(T)

                # We view spheres as wireframe.
                node_sphere = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_sphere)
                node_sphere.paint_uniform_color(const_colors[0])
                sphere_meshes.append(node_sphere)

            # Merge sphere meshes.
            merged_spheres = merge_line_sets(sphere_meshes)
            # merged_spheres = rotate_around_axis(merged_spheres, axis_name="x", angle=-np.pi)

            self.node_meshes.append(merged_spheres)

            # Generate edge meshes.
            K = self.num_neighbors
            min_neighbor_weight = self.edge_weight_threshold
            if self.viz_edges:
                points = np.zeros((cfg.num_nodes, 3))
                edge_coords = []
                for node_id in range(cfg.num_nodes):
                    # Compute nearest K neighbors.
                    max_idxs = np.argpartition(affinity_matrix[node_id], -K)[-K:]

                    for max_idx in max_idxs:
                        max_val = affinity_matrix[node_id, max_idx]

                        source_idx = node_id
                        target_idx = max_idx

                        if math.isfinite(max_val) and max_val >= min_neighbor_weight:
                            edge_coords.append([source_idx, target_idx])

                    # Store center position.
                    center = centers[0, node_id].cpu().numpy()
                    points[node_id] = center

                if len(edge_coords) > 0:
                    line_mesh = LineMesh(points, edge_coords, radius=0.005)
                    line_meshes = merge_meshes(line_mesh.get_line_meshes())

                    # Transform to OpenGL coordinates
                    # line_meshes = rotate_around_axis(line_meshes, axis_name="x", angle=-np.pi)

                    self.edge_meshes.append(line_meshes)

                else:
                    self.edge_meshes.append(None)

        ###############################################################################################
        # Generate reconstructed meshes.
        ###############################################################################################
        self.obj_meshes = []

        if not only_graph_model and not self.viz_only_graph:
            print("Sampling SDF values and extracting meshes ...")

            dim: int = self.grid_dim
            num_chunks_mlp = self.grid_num_chunks
            num_chunks_weights = 128
            grid_size = 0.7

            influence_threshold = 0.02
            print("Influence threshold: {}".format(influence_threshold))

            # Sample grid points
            points = sample_grid_points(dim, grid_size)
            points = torch.from_numpy(np.transpose(points.reshape(3, -1), (1, 0))).cuda()

            num_points = points.shape[0]
            assert num_points % num_chunks_mlp == 0, "The number of points in the grid must be divisible by the number of chunks"
            points_per_chunk_mlp = int(num_points / num_chunks_mlp)
            print("Num. points per chunk: {}".format(points_per_chunk_mlp))

            for t in tqdm(range(len(time_steps))):
                torch.cuda.empty_cache()

                with torch.no_grad():
                    # Move to device
                    grid = torch.from_numpy(self.grid_array[t]).cuda().unsqueeze(0)
                    rotated2gaps = torch.from_numpy(self.rotated2gaps_array[t]).cuda().unsqueeze(0)
                    world2grid = torch.from_numpy(self.world2grid_array[t]).cuda().unsqueeze(0)

                    # Compute augmented inputs.
                    sdfs = augment_grid(grid, world2grid, rotated2gaps)

                    # Predict reconstruction.
                    sdf_pred = np.empty((num_points), dtype=np.float32)

                    for i in range(num_chunks_mlp):
                        points_i = points[i * points_per_chunk_mlp:(i + 1) * points_per_chunk_mlp, :].unsqueeze(0)

                        sdf_pred_i = model(points_i, sdfs, rotated2gaps)

                        sdf_pred[i * points_per_chunk_mlp:(i + 1) * points_per_chunk_mlp] = sdf_pred_i.cpu().numpy()

                    # Extract mesh with Marching cubes.
                    sdf_pred = sdf_pred.reshape(dim, dim, dim)
                    vertices, triangles = mcubes.marching_cubes(sdf_pred, 0)

                    if vertices.shape[0] > 0 and triangles.shape[0] > 0:
                        # Normalize vertices to be in [-grid_size, grid_size]
                        vertices = 2.0 * grid_size * (vertices / (dim - 1)) - grid_size

                        # Convert extracted surface to o3d mesh.
                        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                                         o3d.utility.Vector3iVector(triangles))
                        mesh.compute_vertex_normals()

                        # Transform to OpenGL coordinates
                        # mesh = rotate_around_axis(mesh, axis_name="x", angle=-np.pi)
                    else:
                        print("No mesh vertices are extracted!")
                        mesh = None  # o3d.geometry.TriangleMesh.create_sphere(radius=0.5)

                self.obj_meshes.append(mesh)

        else:
            print("Only graph model was provided, so no reconstruction is executed!")
            print("Graph will be visualized together with ground truth meshes.")

            for gt_mesh in self.gt_meshes:
                self.obj_meshes.append(copy.deepcopy(gt_mesh))

    def _update_obj(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        prev_obj_mesh = self.obj_mesh
        self.obj_mesh = self.obj_meshes[self.time]

        if self.show_spheres and self.obj_mesh is not None:
            # Convert to wireframe.
            self.obj_mesh = o3d.geometry.LineSet.create_from_triangle_mesh(self.obj_mesh)

        if self.obj_mesh is not None:
            vis.add_geometry(self.obj_mesh)

        if prev_obj_mesh is not None:
            vis.remove_geometry(prev_obj_mesh)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def _update_gt(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        prev_gt_mesh = self.gt_mesh

        # If requested, we show a (new) mesh.
        if self.show_gt and len(self.gt_meshes) > 0:
            self.gt_mesh = self.gt_meshes[self.time]
            vis.add_geometry(self.gt_mesh)
        else:
            self.gt_mesh = None

        if prev_gt_mesh is not None:
            vis.remove_geometry(prev_gt_mesh)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def _update_spheres(self, vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()

        if self.sphere_mesh is not None:
            vis.remove_geometry(self.sphere_mesh)
            self.sphere_mesh = None

        if self.edge_mesh is not None:
            vis.remove_geometry(self.edge_mesh)
            self.edge_mesh = None

        if self.show_spheres:
            gt_mesh_idx = self.time

            self.sphere_mesh = self.node_meshes[self.time]
            vis.add_geometry(self.sphere_mesh)

            if self.viz_edges:
                self.edge_mesh = self.edge_meshes[self.time]

                if self.edge_mesh is not None: vis.add_geometry(self.edge_mesh)

        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(param)

    def run(self):
        # Define callbacks.
        def toggle_next(vis):
            # print("::toggle_next")
            self.time += 1
            if self.time >= len(self.obj_meshes):
                self.time = 0

            self._update_obj(vis)
            self._update_gt(vis)
            self._update_spheres(vis)

            # print(f"frame: {self.time * self.time_inc}")

            return False

        def toggle_previous(vis):
            # print("::toggle_previous")
            self.time -= 1
            if self.time < 0:
                self.time = len(self.obj_meshes) - 1

            self._update_obj(vis)
            self._update_gt(vis)
            self._update_spheres(vis)

            # print(f"frame: {self.time * self.time_inc}")

            return False

        def toggle_groundtruth(vis):
            # print("::toggle_groundtruth")
            self.show_gt = not self.show_gt

            self._update_gt(vis)

            return False

        def toggle_spheres(vis):
            # print("::toggle_spheres")
            self.show_spheres = not self.show_spheres

            self._update_obj(vis)
            self._update_spheres(vis)

            return False

        key_to_callback = {}
        key_to_callback[ord("D")] = toggle_next
        key_to_callback[ord("A")] = toggle_previous
        key_to_callback[ord("G")] = toggle_groundtruth
        key_to_callback[ord("N")] = toggle_spheres

        # Add mesh at initial time step.
        assert self.time < len(self.obj_meshes)
        self.obj_mesh = self.obj_meshes[self.time]

        if not self.obj_mesh:
            print("Object mesh doesn't exist. Exiting ...")
            exit(1)

        # Print instructions.
        print()
        print("#" * 100)
        print("VISUALIZATION CONTROLS")
        print("#" * 100)
        print()
        print("N: toggle graph nodes and edges")
        print("G: toggle ground truth")
        print("D: show next")
        print("A: show previous")
        print("S: toggle smooth shading")
        print()

        # Run visualization.
        o3d.visualization.draw_geometries_with_key_callbacks([self.obj_mesh], key_to_callback)


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', '--checkpoint_path', action='store', dest='checkpoint_path')
    parser.add_argument('--time_inc', type=check_non_negative)
    parser.add_argument('--gt_data_dir', action='store', dest='gt_data_dir')
    parser.add_argument('--grid_dim', choices=[32, 64, 128, 256], type=int, help='Grid dimension')
    parser.add_argument('--grid_num_chunks', type=check_positive, help='Number of grid chunks')
    parser.add_argument('--num_neighbors', type=check_non_negative, help='Number of visualized graph neighbors')
    parser.add_argument('--edge_weight_threshold', type=float, help='Graph edge weight threshold')
    parser.add_argument('--viz_only_graph', action='store_true',
                        help='Specify if you want to visualize only graph (much faster)')

    args = parser.parse_args()

    viewer = Viewer(
        args.checkpoint_path,
        time_inc=args.time_inc, gt_data_dir=args.gt_data_dir,
        grid_dim=args.grid_dim, grid_num_chunks=args.grid_num_chunks,
        num_neighbors=args.num_neighbors, edge_weight_threshold=args.edge_weight_threshold,
        viz_only_graph=args.viz_only_graph
    )
    viewer.run()
