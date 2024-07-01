# Modified by 2024 Allen Ren, Princeton University
# Copyright (c) 2018 Andy Zeng
# Source: https://github.com/andyzeng/tsdf-fusion-python/blob/master/fusion.py
# BSD 2-Clause License
# Copyright (c) 2019, Princeton University
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from numba import njit, prange
import random
import copy
import logging
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from skimage import measure
from sklearn.cluster import DBSCAN, KMeans
from scipy.ndimage import gaussian_filter
from .geom import *
from .habitat import pos_normal_to_habitat, pos_habitat_to_normal
import habitat_sim
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import supervision as sv


@dataclass
class Frontier:
    """Frontier class for frontier-based exploration."""

    position: np.ndarray  # integer position in voxel grid
    orientation: np.ndarray  # directional vector of the frontier in float
    region: np.ndarray  # boolean array of the same shape as the voxel grid, indicating the region of the frontier
    frontier_id: int  # unique id for the frontier to identify its region on the frontier map
    is_stuck: bool = False  # whether the frontier has been visited before
    image: str = None

    def __eq__(self, other):
        if not isinstance(other, Frontier):
            raise TypeError("Cannot compare Frontier with non-Frontier object.")
        return np.array_equal(self.region, other.region)


@dataclass
class Object:
    """Object class for semantic objects."""

    position: np.ndarray  # integer position in voxel grid
    object_id: int


class TSDFPlanner:
    """Volumetric TSDF Fusion of RGB-D Images. No GPU mode.

    Add frontier-based exploration and semantic map.
    """

    def __init__(
        self,
        vol_bnds,
        voxel_size,
        floor_height_offset=0,
        pts_init=None,
        init_clearance=0,
    ):
        """Constructor.
        Args:
          vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
            xyz bounds (min/max) in meters.
          voxel_size (float): The volume discretization in meters.
        """
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."
        assert (vol_bnds[:, 0] < vol_bnds[:, 1]).all()

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = 5 * self._voxel_size  # truncation on SDF
        self._color_const = 256 * 256

        # Adjust volume bounds and ensure C-order contiguous
        self._vol_dim = (
            np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size)
            .copy(order="C")
            .astype(int)
        )
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order="C").astype(np.float32)

        # Initialize pointers to voxel volume in CPU memory
        # Assume all unobserved regions are occupied
        self._tsdf_vol_cpu = -np.ones(self._vol_dim).astype(np.float32)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        # Semantic value
        self._val_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._weight_val_vol_cpu = np.zeros(self._vol_dim[:2]).astype(np.float32)

        # Explored or not
        self._explore_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        # Get voxel grid coordinates
        xv, yv, zv = np.meshgrid(
            range(self._vol_dim[0]),
            range(self._vol_dim[1]),
            range(self._vol_dim[2]),
            indexing="ij",
        )
        self.vox_coords = (
            np.concatenate(
                [xv.reshape(1, -1), yv.reshape(1, -1), zv.reshape(1, -1)], axis=0
            )
            .astype(int)
            .T
        )

        # pre-compute
        self.cam_pts_pre = TSDFPlanner.vox2world(
            self._vol_origin, self.vox_coords, self._voxel_size
        )

        # Find the minimum height voxel
        self.min_height_voxel = int(floor_height_offset / self._voxel_size)

        # For masking the area around initial pose to be unoccupied
        coords_init = self.world2vox(pts_init)
        self.init_points = points_in_circle(
            coords_init[0],
            coords_init[1],
            int(init_clearance / self._voxel_size),
            self._vol_dim[:2],
        )

        self.simple_scene_graph = {}
        self.frontiers: List[Frontier] = []

        # about frontiers
        self.frontier_map = np.zeros(self._vol_dim[:2], dtype=int)
        self.frontier_counter = 1

        self.unexplored = None
        self.unoccupied = None
        self.occupied = None
        self.island = None
        self.unexplored_neighbors = None
        self.occupied_map_camera = None

        self.frontiers_weight = None

    def update_scene_graph(self, detection_model, rgb, semantic_obs, obj_id_to_name, obj_id_to_bbox, cfg, target_obj_id, return_annotated=False):
        target_found = False

        unique_obj_ids = np.unique(semantic_obs)
        class_to_obj_id = {}
        for obj_id in unique_obj_ids:
            if obj_id == 0 or obj_id not in obj_id_to_name.keys() or obj_id_to_name[obj_id] in ['wall', 'floor', 'ceiling']:
                continue
            if obj_id_to_name[obj_id] not in class_to_obj_id.keys():
                class_to_obj_id[obj_id_to_name[obj_id]] = [obj_id]
            else:
                class_to_obj_id[obj_id_to_name[obj_id]].append(obj_id)
        all_classes = list(class_to_obj_id.keys())

        if len(all_classes) == 0:
            if return_annotated:
                return target_found, rgb
            else:
                return target_found

        detection_model.set_classes(all_classes)

        results = detection_model.infer(rgb, confidence=cfg.confidence)
        detections = sv.Detections.from_inference(results).with_nms(threshold=cfg.nms_threshold)

        adopted_indices = []
        for i in range(len(detections)):
            class_name = all_classes[detections.class_id[i]]
            x_start, y_start, x_end, y_end = detections.xyxy[i].astype(int)
            bbox_mask = np.zeros(semantic_obs.shape, dtype=bool)
            bbox_mask[y_start:y_end, x_start:x_end] = True
            for obj_id in class_to_obj_id[class_name]:

                if obj_id not in obj_id_to_bbox.keys():
                    continue

                obj_x_start, obj_y_start = np.argwhere(semantic_obs == obj_id).min(axis=0)
                obj_x_end, obj_y_end = np.argwhere(semantic_obs == obj_id).max(axis=0)
                obj_mask = np.zeros(semantic_obs.shape, dtype=bool)
                obj_mask[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = True
                if IoU(bbox_mask, obj_mask) > cfg.iou_threshold:
                    # this object is counted as detected
                    # add to the scene graph
                    if obj_id not in self.simple_scene_graph.keys():
                        bbox = obj_id_to_bbox[obj_id]["bbox"]
                        bbox = np.asarray(bbox)
                        bbox_center = np.mean(bbox, axis=0)
                        # change to x, z, y for habitat
                        bbox_center = bbox_center[[0, 2, 1]]
                        # add to simple scene graph
                        self.simple_scene_graph[obj_id] = bbox_center

                    if obj_id == target_obj_id:
                        target_found = True
                        adopted_indices.append(i)

                    break

        if return_annotated:
            if len(adopted_indices) == 0:
                return target_found, rgb

            # filter out the detections that are not adopted
            detections = detections[adopted_indices]

            annotated_image = rgb.copy()
            BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
            LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)
            annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
            annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
            return target_found, annotated_image

    @staticmethod
    @njit(parallel=True)
    def vox2world(vol_origin, vox_coords, vox_size):
        """Convert voxel grid coordinates to world coordinates."""
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    @staticmethod
    @njit(parallel=True)
    def cam2pix(cam_pts, intr):
        """Convert camera coordinates to pixel coordinates."""
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix

    def pix2cam(self, pix, intr):
        """Convert pixel coordinates to camera coordinates."""
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        cam_pts = np.empty((pix.shape[0], 3), dtype=np.float32)
        for i in range(cam_pts.shape[0]):
            cam_pts[i, 2] = 1
            cam_pts[i, 0] = (pix[i, 0] - cx) / fx * cam_pts[i, 2]
            cam_pts[i, 1] = (pix[i, 1] - cy) / fy * cam_pts[i, 2]
        return cam_pts

    def world2vox(self, pts):
        pts = pts - self._vol_origin
        coords = np.round(pts / self._voxel_size).astype(int)
        coords = np.clip(coords, 0, self._vol_dim - 1)
        return coords

    @staticmethod
    @njit(parallel=True)
    def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
        """Integrate the TSDF volume."""
        tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in prange(len(tsdf_vol)):
            w_new[i] = w_old[i] + obs_weight
            tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
        return tsdf_vol_int, w_new

    def integrate_sem(
        self,
        sem_pix,
        radius=1.0,  # meter
        obs_weight=1.0,
    ):
        """Add semantic value to the 2D map by marking a circle of specified radius"""
        assert len(self.candidates) == len(sem_pix)
        for p_ind, p in enumerate(self.candidates):
            radius_vox = int(radius / self._voxel_size)
            pts = points_in_circle(p[0], p[1], radius_vox, self._vol_dim[:2])
            for pt in pts:
                w_old = self._weight_val_vol_cpu[pt[0], pt[1]].copy()
                self._weight_val_vol_cpu[pt[0], pt[1]] += obs_weight
                self._val_vol_cpu[pt[0], pt[1]] = (
                    w_old * self._val_vol_cpu[pt[0], pt[1]]
                    + obs_weight * sem_pix[p_ind]
                ) / self._weight_val_vol_cpu[pt[0], pt[1]]

    def integrate(
        self,
        color_im,
        depth_im,
        cam_intr,
        cam_pose,
        sem_im=None,
        w_new=None,
        obs_weight=1.0,
        margin_h=240,  # from top
        margin_w=120,  # each side
    ):
        """Integrate an RGB-D frame into the TSDF volume.
        Args:
          color_im (ndarray): An RGB image of shape (H, W, 3).
          depth_im (ndarray): A depth image of shape (H, W).
          cam_intr (ndarray): The camera intrinsics matrix of shape (3, 3).
          cam_pose (ndarray): The camera pose (i.e. extrinsics) of shape (4, 4).
          sem_im (ndarray): An semantic image of shape (H, W).
          obs_weight (float): The weight to assign for the current observation. A higher
            value
          margin_h (int): The margin from the top of the image to exclude when integrating explored
          margin_w (int): The margin from the sides of the image to exclude when integrating explored
        """
        im_h, im_w = depth_im.shape

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(
            color_im[..., 2] * self._color_const
            + color_im[..., 1] * 256
            + color_im[..., 0]
        )

        # Convert voxel grid coordinates to pixel coordinates
        cam_pts = rigid_transform(self.cam_pts_pre, np.linalg.inv(cam_pose))
        pix_z = cam_pts[:, 2]
        pix = TSDFPlanner.cam2pix(cam_pts, cam_intr)
        pix_x, pix_y = pix[:, 0], pix[:, 1]

        # Eliminate pixels outside view frustum
        valid_pix = np.logical_and(
            pix_x >= 0,
            np.logical_and(
                pix_x < im_w,
                np.logical_and(pix_y >= 0, np.logical_and(pix_y < im_h, pix_z > 0)),
            ),
        )
        depth_val = np.zeros(pix_x.shape)
        depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

        # narrow view
        valid_pix_narrow = np.logical_and(
            pix_x >= margin_w,
            np.logical_and(
                pix_x < (im_w - margin_w),
                np.logical_and(
                    pix_y >= margin_h,
                    np.logical_and(pix_y < im_h, pix_z > 0),
                ),
            ),
        )
        depth_val_narrow = np.zeros(pix_x.shape)
        depth_val_narrow[valid_pix_narrow] = depth_im[
            pix_y[valid_pix_narrow], pix_x[valid_pix_narrow]
        ]

        # Integrate TSDF
        depth_diff = depth_val - pix_z
        valid_pts = np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin)
        dist = np.maximum(-1, np.minimum(1, depth_diff / self._trunc_margin))
        valid_vox_x = self.vox_coords[valid_pts, 0]
        valid_vox_y = self.vox_coords[valid_pts, 1]
        valid_vox_z = self.vox_coords[valid_pts, 2]
        w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]

        depth_diff_narrow = depth_val_narrow - pix_z
        valid_pts_narrow = np.logical_and(
            depth_val_narrow > 0, depth_diff_narrow >= -self._trunc_margin
        )
        valid_vox_x_narrow = self.vox_coords[valid_pts_narrow, 0]
        valid_vox_y_narrow = self.vox_coords[valid_pts_narrow, 1]
        valid_vox_z_narrow = self.vox_coords[valid_pts_narrow, 2]
        if w_new is None:
            tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            valid_dist = dist[valid_pts]
            tsdf_vol_new, w_new = TSDFPlanner.integrate_tsdf(
                tsdf_vals, valid_dist, w_old, obs_weight
            )
            self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
            self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

            # Mark explored
            self._explore_vol_cpu[
                valid_vox_x_narrow, valid_vox_y_narrow, valid_vox_z_narrow
            ] = 1

            # Integrate color
            old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            old_b = np.floor(old_color / self._color_const)
            old_g = np.floor((old_color - old_b * self._color_const) / 256)
            old_r = old_color - old_b * self._color_const - old_g * 256
            new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_b = np.floor(new_color / self._color_const)
            new_g = np.floor((new_color - new_b * self._color_const) / 256)
            new_r = new_color - new_b * self._color_const - new_g * 256
            new_b = np.minimum(
                255.0, np.round((w_old * old_b + obs_weight * new_b) / w_new)
            )
            new_g = np.minimum(
                255.0, np.round((w_old * old_g + obs_weight * new_g) / w_new)
            )
            new_r = np.minimum(
                255.0, np.round((w_old * old_r + obs_weight * new_r) / w_new)
            )
            self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = (
                new_b * self._color_const + new_g * 256 + new_r
            )

        # Integrate semantics if specified
        if sem_im is not None:
            old_sem = self._val_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
            new_sem = sem_im[pix_y[valid_pts], pix_x[valid_pts]]
            new_sem = (w_old * old_sem + obs_weight * new_sem) / w_new
            self._val_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_sem
        return w_new

    def get_volume(self):
        return self._tsdf_vol_cpu, self._color_vol_cpu

    def get_point_cloud(self):
        """Extract a point cloud from the voxel volume."""
        tsdf_vol, color_vol = self.get_volume()

        # Marching cubes
        # verts = measure.marching_cubes(tsdf_vol, level=0, method='lewiner')[0]
        # See: https://github.com/andyzeng/tsdf-fusion-python/issues/24
        verts = measure.marching_cubes(
            tsdf_vol, mask=np.logical_and(tsdf_vol > -0.5, tsdf_vol < 0.5), level=0
        )[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        pc = np.hstack([verts, colors])
        return pc

    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes."""
        tsdf_vol, color_vol = self.get_volume()

        # Marching cubes
        # verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0, method='lewiner')
        # See: https://github.com/andyzeng/tsdf-fusion-python/issues/24
        verts, faces, norms, vals = measure.marching_cubes(
            tsdf_vol, mask=np.logical_and(tsdf_vol > -0.5, tsdf_vol < 0.5), level=0
        )
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b * self._color_const) / 256)
        colors_r = rgb_vals - colors_b * self._color_const - colors_g * 256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors

    ############# For building semantic map and exploration #############

    def update_frontier_map(
            self,
            pts,
            cfg
    ) -> bool:
        """Determine the next frontier to traverse to with semantic-value-weighted sampling."""
        cur_point = self.world2vox(pts)

        island, unoccupied = self.get_island_around_pts(pts, height=0.4)
        occupied = np.logical_not(unoccupied).astype(int)
        unexplored = (np.sum(self._explore_vol_cpu, axis=-1) == 0).astype(int)
        for point in self.init_points:
            unexplored[point[0], point[1]] = 0
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        unexplored_neighbors = ndimage.convolve(
            unexplored, kernel, mode="constant", cval=0.0
        )
        occupied_map_camera = np.logical_not(
            self.get_island_around_pts(pts, height=1.2)[0]
        )
        self.unexplored = unexplored
        self.unoccupied = unoccupied
        self.occupied = occupied
        self.island = island
        self.unexplored_neighbors = unexplored_neighbors
        self.occupied_map_camera = occupied_map_camera

        # detect and update frontiers
        frontier_areas = np.argwhere(
            island
            & (unexplored_neighbors >= cfg.frontier_area_min)
            & (unexplored_neighbors <= cfg.frontier_area_max)
        )
        frontier_edge_areas = np.argwhere(
            island
            & (unexplored_neighbors >= cfg.frontier_edge_area_min)
            & (unexplored_neighbors <= cfg.frontier_edge_area_max)
        )

        if len(frontier_areas) == 0:
            # this happens when there are stairs on the floor, and the planner cannot handle this situation
            # just skip this question
            logging.error(f'Error in find_next_pose_with_path: frontier area size is 0')
            return False

        # cluster frontier regions
        db = DBSCAN(eps=cfg.eps, min_samples=2).fit(frontier_areas)
        labels = db.labels_
        # get one point from each cluster
        valid_ft_angles = []
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster = frontier_areas[labels == label]

            # filter out small frontiers
            area = len(cluster)
            if area < cfg.min_frontier_area:
                continue

            # convert the cluster from voxel coordinates to polar angle coordinates
            angle_cluster = np.asarray([
                np.arctan2(cluster[i, 1] - cur_point[1], cluster[i, 0] - cur_point[0])
                for i in range(len(cluster))
            ])  # range from -pi to pi

            # get the range of the angles
            angle_range = get_angle_span(angle_cluster)
            warping_gap = get_warping_gap(angle_cluster)  # add 2pi to angles that smaller than this to avoid angles crossing -pi/pi line
            if warping_gap is not None:
                angle_cluster[angle_cluster < warping_gap] += 2 * np.pi

            if angle_range > cfg.max_frontier_angle_range_deg * np.pi / 180:
                # cluster again on the angle, ie, split the frontier
                num_clusters = int(angle_range / (cfg.max_frontier_angle_range_deg * np.pi / 180)) + 1
                db_angle = KMeans(n_clusters=num_clusters).fit(angle_cluster[..., None])
                labels_angle = db_angle.labels_
                for label_angle in np.unique(labels_angle):
                    if label_angle == -1:
                        continue
                    ft_angle = np.mean(angle_cluster[labels_angle == label_angle])
                    valid_ft_angles.append({
                        'angle': ft_angle - 2 * np.pi if ft_angle > np.pi else ft_angle,
                        'region': self.get_frontier_region_map(cluster[labels_angle == label_angle]),
                    })
            else:
                ft_angle = np.mean(angle_cluster)
                valid_ft_angles.append({
                    'angle': ft_angle - 2 * np.pi if ft_angle > np.pi else ft_angle,
                    'region': self.get_frontier_region_map(cluster),
                })

        # remove frontiers that have been changed
        filtered_frontiers = []
        kept_frontier_area = np.zeros_like(self.frontier_map, dtype=bool)
        for frontier in self.frontiers:
            if frontier in filtered_frontiers:
                continue
            IoU_values = np.asarray([IoU(frontier.region, new_ft['region']) for new_ft in valid_ft_angles])
            pix_diff_values = np.asarray([pix_diff(frontier.region, new_ft['region']) for new_ft in valid_ft_angles])
            print(f'IoU values: {IoU_values}')
            frontier_appended = False
            if np.any((IoU_values > cfg.region_equal_threshold) | (pix_diff_values <= 3)):
                # the frontier is not changed (almost)
                filtered_frontiers.append(frontier)
                kept_frontier_area = kept_frontier_area | frontier.region
                frontier_appended = True
                # then remove that new frontier
                ft_idx = np.argmax(IoU_values)
                valid_ft_angles.pop(ft_idx)
            elif np.sum(IoU_values > 0.02) >= 2 and cfg.region_equal_threshold < np.sum(IoU_values[IoU_values > 0.02]) <= 1:
                # if one old frontier is split into two new frontiers, and their sizes are equal
                # then keep the old frontier
                logging.debug(f"Frontier one split to many: {IoU_values[IoU_values > 0.02]}")
                filtered_frontiers.append(frontier)
                kept_frontier_area = kept_frontier_area | frontier.region
                frontier_appended = True
                # then remove those new frontiers
                ft_ids = list(np.argwhere(IoU_values > 0.02).squeeze())
                ft_ids.sort(reverse=True)
                for ft_idx in ft_ids:
                    valid_ft_angles.pop(ft_idx)
            elif np.sum(IoU_values > 0.02) == 1:
                # if some old frontiers are merged into one new frontier
                ft_idx = np.argmax(IoU_values)
                IoU_with_old_ft = np.asarray([IoU(valid_ft_angles[ft_idx]['region'], ft.region) for ft in self.frontiers])
                print(f'IoU with old frontiers: {IoU_with_old_ft}')
                if np.sum(IoU_with_old_ft > 0.02) >= 2 and cfg.region_equal_threshold < np.sum(IoU_with_old_ft[IoU_with_old_ft > 0.02]) <= 1:
                    # if the new frontier is merged from two or more old frontiers, and their sizes are equal
                    # then add all the old frontiers
                    logging.debug(f"Frontier many merged to one: {IoU_with_old_ft[IoU_with_old_ft > 0.02]}")
                    for i in list(np.argwhere(IoU_with_old_ft > 0.02).squeeze()):
                        if self.frontiers[i] not in filtered_frontiers:
                            filtered_frontiers.append(self.frontiers[i])
                            kept_frontier_area = kept_frontier_area | self.frontiers[i].region
                    valid_ft_angles.pop(ft_idx)
                    frontier_appended = True

            if not frontier_appended:
                self.free_frontier(frontier)
                if np.any(IoU_values > 0.8):
                    # the frontier is slightly updated
                    # choose the new frontier that updates the current frontier
                    update_ft_idx = np.argmax(IoU_values)
                    ang = valid_ft_angles[update_ft_idx]['angle']
                    # if the new frontier has no valid observations
                    if 1 > self._voxel_size * get_collision_distance(
                        occupied_map=occupied_map_camera,
                        pos=cur_point,
                        direction=np.array([np.cos(ang), np.sin(ang)])
                    ):
                        # create a new frontier with the old image
                        old_img_path = frontier.image
                        filtered_frontiers.append(
                            self.create_frontier(valid_ft_angles[update_ft_idx], frontier_edge_areas=frontier_edge_areas, cur_point=cur_point)
                        )
                        filtered_frontiers[-1].image = old_img_path
                        valid_ft_angles.pop(update_ft_idx)
                        kept_frontier_area = kept_frontier_area | filtered_frontiers[-1].region
        self.frontiers = filtered_frontiers

        # merge new frontiers if they are too close
        if len(valid_ft_angles) > 1:
            valid_ft_angles_new = []
            # sort the angles
            valid_ft_angles = sorted(valid_ft_angles, key=lambda x: x['angle'])
            while len(valid_ft_angles) > 0:
                cur_angle = valid_ft_angles.pop(0)
                if len(valid_ft_angles) > 0:
                    next_angle = valid_ft_angles[0]
                    if next_angle['angle'] - cur_angle['angle'] < cfg.min_frontier_angle_diff_deg * np.pi / 180:
                        # merge the two
                        weight = np.sum(cur_angle['region']) / (np.sum(cur_angle['region']) + np.sum(next_angle['region']))
                        cur_angle['angle'] = cur_angle['angle'] * weight + next_angle['angle'] * (1 - weight)
                        cur_angle['region'] = cur_angle['region'] | next_angle['region']
                        valid_ft_angles.pop(0)
                valid_ft_angles_new.append(cur_angle)
            valid_ft_angles = valid_ft_angles_new

        # create new frontiers and add to frontier list
        for ft_data in valid_ft_angles:
            region_covered_ratio = np.sum(ft_data['region'] & kept_frontier_area) / np.sum(ft_data['region'])
            if region_covered_ratio < 1 - cfg.region_equal_threshold:
                # if this frontier is not covered by the existing frontiers, then add it
                self.frontiers.append(
                    self.create_frontier(ft_data, frontier_edge_areas=frontier_edge_areas, cur_point=cur_point)
                )

        return True

    def get_next_choice(
            self,
            pts,
            angle,
            path_points,
            pathfinder,
            target_obj_id,
            cfg
    ) -> Optional[Union[Object, Frontier]]:
        cur_point = self.world2vox(pts)
        
        # determine whether the target object is in scene graph
        if target_obj_id in self.simple_scene_graph.keys():
            target_point = self.habitat2voxel(self.simple_scene_graph[target_obj_id])[:2]
            logging.info(f"Next choice: Object at {target_point}")
            self.frontiers_weight = np.zeros((len(self.frontiers)))
            return Object(target_point.astype(int), target_obj_id)
        else:
            frontiers_weight = np.empty(0)
            for frontier in self.frontiers:
                # find normal of the frontier
                normal = frontier.orientation

                # Then check how much unoccupied in that direction
                max_pixel_check = int(cfg.max_unoccupied_check_frontier / self._voxel_size)
                dir_pts = np.round(
                    frontier.position + np.arange(max_pixel_check)[:, np.newaxis] * normal
                ).astype(int)
                dir_pts = self.clip_2d_array(dir_pts)
                unoccupied_rate = (
                    np.sum(self.unoccupied[dir_pts[:, 0], dir_pts[:, 1]] == 1)
                    / max_pixel_check
                )

                # Check the ratio of unexplored in the direction, until hits obstacle
                max_pixel_check = int(cfg.max_unexplored_check_frontier / self._voxel_size)
                dir_pts = np.round(
                    frontier.position + np.arange(max_pixel_check)[:, np.newaxis] * normal
                ).astype(int)
                dir_pts = self.clip_2d_array(dir_pts)
                unexplored_rate = (
                    np.sum(self.unexplored[dir_pts[:, 0], dir_pts[:, 1]] == 1)
                    / max_pixel_check
                )

                # get weight for path points
                pos_world = frontier.position * self._voxel_size + self._vol_origin[:2]
                closest_dist, cosine_dist = self.get_closest_distance(path_points, pos_world, normal, pathfinder, pts[2])

                # Get weight - unexplored, unoccupied, and value
                weight = np.exp(unexplored_rate / cfg.unexplored_T)  # [0-1] before T
                weight *= np.exp(unoccupied_rate / cfg.unoccupied_T)  # [0-1] before T
                
                # add weight for path points
                weight *= np.exp(- closest_dist) * 3
                weight *= np.exp(cosine_dist)

                # Check distance to current point - make weight very small if too close and aligned
                dist = np.sqrt((cur_point[0] - frontier.position[0]) ** 2 + (cur_point[1] - frontier.position[1]) ** 2)
                pts_angle = np.arctan2(normal[1], normal[0]) - np.pi / 2
                weight *= np.exp(-dist / cfg.dist_T)
                if (
                    dist < cfg.min_dist_from_cur / self._voxel_size
                    and np.abs(angle - pts_angle) < np.pi / 6
                ):
                    weight *= 1e-3

                # if the frontier is stuck, then reduce the weight
                if frontier.is_stuck:
                    weight *= 1e-3

                # Save weight
                frontiers_weight = np.append(frontiers_weight, weight)
            logging.info(f"Number of frontiers for next pose: {len(self.frontiers)}")
            self.frontiers_weight = frontiers_weight

            # choose the frontier with highest weight
            if len(self.frontiers) > 0:
                frontier_ind = np.argmax(frontiers_weight)
                logging.info(f"Next choice: Frontier at {self.frontiers[frontier_ind].position} with weight {frontiers_weight[frontier_ind]:.3f}")
                return self.frontiers[frontier_ind]
            else:
                logging.error(f"Error in get_next_choice: no frontiers")
                return None
    
    def get_next_navigation_point(
        self,
        choice: Union[Object, Frontier],
        pts,
        angle,
        path_points,
        pathfinder,
        cfg,
        save_visualization=True,
    ):
        cur_point = self.world2vox(pts)
        max_point = choice

        if type(choice) == Object:
            target_point = choice.position
            # # set the object center as the navigation target
            # target_navigable_point = get_nearest_true_point(target_point, unoccupied)  # get the nearest unoccupied point for the nav target
            # since it's not proper to directly go to the target point,
            # we'd better find a navigable point that is certain distance from it to better observe the target
            target_navigable_point = get_proper_observe_point(target_point, self.unoccupied, cur_point=cur_point , dist=cfg.final_observe_distance / self._voxel_size)
            if target_navigable_point is None:
                # this is usually because the target object is too far, so its surroundings are not detected as unoccupied
                # so we just temporarily use pathfinder to find a navigable point around it
                target_point_normal = target_point * self._voxel_size + self._vol_origin[:2]
                target_point_normal = np.append(target_point_normal, pts[-1])
                target_point_habitat = pos_normal_to_habitat(target_point_normal)
                try_count = 0
                while True:
                    try_count += 1
                    if try_count > 100:
                        logging.error(f"Error in find_next_pose_with_path: cannot find a proper next point near object at {target_point}")
                        return (None,)
                    try:
                        target_navigable_point_habitat = pathfinder.get_random_navigable_point_near(
                            circle_center=target_point_habitat,
                            radius=1.0,
                        )
                    except:
                        logging.error(f"Error in find_next_pose_with_path: pathfinder.get_random_navigable_point_near failed")
                        continue
                    if np.isnan(target_navigable_point_habitat).any():
                        logging.error(f"Error in find_next_pose_with_path: pathfinder.get_random_navigable_point_near returned nan")
                        continue
                    if abs(target_navigable_point_habitat[1] - pts[-1]) < 0.1:
                        break
                target_navigable_point = self.habitat2voxel(target_navigable_point_habitat)[:2]
            next_point = target_navigable_point
        elif type(choice) == Frontier:
            # find the direction into unexplored
            ft_direction = max_point.orientation

            # find an unoccupied point between the agent and the frontier
            next_point = np.array(max_point.position, dtype=float)
            try_count = 0
            while (
                not self.check_within_bnds(next_point.astype(int)) or
                self.occupied[int(next_point[0]), int(next_point[1])] or
                not self.island[int(next_point[0]), int(next_point[1])]
            ):
                next_point -= ft_direction
                try_count += 1
                if try_count > 1000:
                    logging.error(f"Error in find_next_pose_with_path: cannot find a proper next point")
                    return (None,)

            next_point = next_point.astype(int)
        else:
            logging.error(f"Error in find_next_pose_with_path: wrong choice type: {type(choice)}")
            return (None,)
        
        # check the distance to next navigation point
        # if the target navigation point is too far
        # then just go to a point between the current point and the target point
        max_dist_from_cur = cfg.max_dist_from_cur_phase_1 if type(max_point) == Frontier else cfg.max_dist_from_cur_phase_2  # in phase 2, the step size should be smaller
        dist, path_to_target = self.get_distance(cur_point[:2], next_point, height=pts[2], pathfinder=pathfinder)

        if dist > max_dist_from_cur:
            target_arrived = False
            if path_to_target is not None:
                # drop the y value of the path to avoid errors when calculating seg_length
                path_to_target = [np.asarray([p[0], 0.0, p[2]]) for p in path_to_target]
                # if the pathfinder find a path, then just walk along the path for max_dist_from_cur distance
                dist_to_travel = max_dist_from_cur
                middle_point_found = False
                for i in range(len(path_to_target) - 1):
                    seg_length = np.linalg.norm(path_to_target[i + 1] - path_to_target[i])
                    if seg_length < dist_to_travel:
                        dist_to_travel -= seg_length
                    else:
                        # find the point on the segment according to the length ratio
                        next_point_habitat = path_to_target[i] + (path_to_target[i + 1] - path_to_target[i]) * dist_to_travel / seg_length
                        next_point = self.world2vox(pos_habitat_to_normal(next_point_habitat))[:2]
                        middle_point_found = True
                        break
                if not middle_point_found:
                    # this is a very rare case that, the sum of the segment lengths is smaller than the dist returned by the pathfinder
                    # and meanwhile the max_dist_from_cur larger than the sum of the segment lengths
                    # resulting that the previous code cannot find a proper point in the middle of the path
                    # in this case, just keep the next point unchanged
                    target_arrived = True
            else:
                # if the pathfinder cannot find a path, then just go to a point between the current point and the target point
                walk_dir = next_point - cur_point[:2]
                walk_dir = walk_dir / np.linalg.norm(walk_dir)
                next_point = cur_point[:2] + walk_dir * max_dist_from_cur / self._voxel_size
                # ensure next point is valid, otherwise go backward a bit
                try_count = 0
                while (
                    not self.check_within_bnds(next_point)
                    or not self.island[int(np.round(next_point[0])), int(np.round(next_point[1]))]
                    or self.occupied[int(np.round(next_point[0])), int(np.round(next_point[1]))]
                ):
                    next_point -= walk_dir
                    try_count += 1
                    if try_count > 1000:
                        logging.error(f"Error in find_next_pose_with_path: cannot find a proper next point")
                        return (None,)
                next_point = np.round(next_point).astype(int)
        else:
            target_arrived = True

        next_point_old = next_point.copy()
        next_point = adjust_navigation_point(next_point, self.occupied, voxel_size=self._voxel_size, max_adjust_distance=0.1)

        # determine the direction
        if target_arrived:  # if the next arriving position is the target point
            if type(max_point) == Frontier:
                direction = self.rad2vector(angle)  # if the target is a frontier, then the agent's orientation does not change
            else:
                direction = max_point.position - cur_point[:2]  # if the target is an object, then the agent should face the object
        else:  # the agent is still on the way to the target point
            direction = next_point - cur_point[:2]
        if np.linalg.norm(direction) < 1e-6:  # this is a rare case that next point is the same as the current point
            # usually this is a problem in the pathfinder
            logging.warning(f"Warning in agent_step: next point is the same as the current point when determining the direction")
            direction = self.rad2vector(angle)
        direction = direction / np.linalg.norm(direction)

        # if the next point is the same as the current point, and the target is a frontier
        # then the frontier is stuck and cannot be reached
        if np.linalg.norm(next_point - cur_point[:2]) < 1e-6 and type(max_point) == Frontier:
            max_point.is_stuck = True

        # Plot
        fig = None
        if save_visualization:
            fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 18))
            agent_orientation = self.rad2vector(angle)
            ax1.imshow(self.unoccupied)
            ax1.scatter(max_point.position[1], max_point.position[0], c="r", s=30, label="max")
            ax1.scatter(cur_point[1], cur_point[0], c="b", s=30, label="current")
            ax1.arrow(cur_point[1], cur_point[0], agent_orientation[1] * 4, agent_orientation[0] * 4, width=0.1, head_width=0.8, head_length=0.8, color='b')
            ax1.scatter(next_point[1], next_point[0], c="g", s=30, label="actual")
            ax1.scatter(next_point_old[1], next_point_old[0], c="y", s=30, label="old")
            # plot all the detected objects
            for obj_center in self.simple_scene_graph.values():
                obj_vox = self.habitat2voxel(obj_center)
                ax1.scatter(obj_vox[1], obj_vox[0], c="w", s=30)
            # plot the target point if found
            if type(max_point) == Object:
                ax1.scatter(max_point.position[1], max_point.position[0], c="r", s=80, label="target")
            ax1.set_title("Unoccupied")

            island_high = np.logical_not(self.occupied_map_camera)
            ax2.imshow(island_high)
            for frontier in self.frontiers:
                if frontier.image is not None:
                    ax2.scatter(frontier.position[1], frontier.position[0], color="g", s=50, alpha=1)
                else:
                    ax2.scatter(frontier.position[1], frontier.position[0], color="r", s=50, alpha=1)
            ax2.scatter(cur_point[1], cur_point[0], c="b", s=30, label="current")
            ax2.set_title("Island")

            ax3.imshow(self.unexplored_neighbors + self.frontier_map)
            for frontier in self.frontiers:
                ax3.scatter(frontier.position[1], frontier.position[0], color="m", s=10, alpha=1)
                normal = frontier.orientation
                dx, dy = normal * 4
                ax3.arrow(frontier.position[1], frontier.position[0], dy, dx, width=0.1, head_width=0.8, head_length=0.8, color='m')
            ax3.scatter(max_point.position[1], max_point.position[0], c="r", s=30, label="max")
            ax3.set_title("Unexplored neighbors")

            im = ax4.imshow(self.unoccupied)
            for frontier in self.frontiers:
                ax4.scatter(frontier.position[1], frontier.position[0], color="white", s=20, alpha=1)
                normal = frontier.orientation
                dx, dy = normal * 4
                ax4.arrow(frontier.position[1], frontier.position[0], dy, dx, width=0.1, head_width=0.8, head_length=0.8, color='white')
            fig.colorbar(im, orientation="vertical", ax=ax4, fraction=0.046, pad=0.04)
            ax4.scatter(max_point.position[1], max_point.position[0], c="r", s=30, label="max")
            ax4.scatter(cur_point[1], cur_point[0], c="b", s=30, label="current")
            ax4.arrow(cur_point[1], cur_point[0], agent_orientation[1] * 4, agent_orientation[0] * 4, width=0.1, head_width=0.8, head_length=0.8, color='b')
            ax4.scatter(next_point[1], next_point[0], c="g", s=30, label="actual")
            ax4.scatter(next_point_old[1], next_point_old[0], c="y", s=30, label="old")
            ax4.quiver(
                next_point[1],
                next_point[0],
                direction[1],
                direction[0],
                color="r",
                scale=5,
                angles="xy",
                alpha=0.2,
            )
            ax4.set_title("Current sem values")

            im = ax5.imshow(self.island)
            ax5.set_title("Path on island")

            frontier_weights = np.zeros_like(self.frontier_map)
            for frontier, weight in zip(self.frontiers, self.frontiers_weight):
                frontier_weights[frontier.position[0], frontier.position[1]] = weight
            im = ax6.imshow(frontier_weights)
            # draw path points
            for i_pp in range(len(path_points) - 1):
                p1 = (path_points[i_pp] - self._vol_origin[:2]) / self._voxel_size
                p2 = (path_points[i_pp + 1] - self._vol_origin[:2]) / self._voxel_size
                ax6.arrow(p1[1], p1[0], p2[1] - p1[1], p2[0] - p1[0], color="r", width=0.1, head_width=0.8, head_length=0.8)

            fig.colorbar(im, orientation="vertical", ax=ax6, fraction=0.046, pad=0.04)
            # ax6.scatter(max_point[1], max_point[0], c="r", s=20, label="max")
            ax6.set_title("Frontier weights")

        # Convert back to world coordinates
        next_point_normal = next_point * self._voxel_size + self._vol_origin[:2]

        # Find the yaw angle again
        next_yaw = np.arctan2(direction[1], direction[0]) - np.pi / 2

        # update the path points
        updated_path_points = self.update_path_points(path_points, next_point_normal)

        return next_point_normal, next_yaw, next_point, fig, updated_path_points

    def get_island_around_pts(self, pts, fill_dim=0.4, height=0.4):
        """Find the empty space around the point (x,y,z) in the world frame"""
        # Convert to voxel coordinates
        cur_point = self.world2vox(pts)

        # Check if the height voxel is occupied
        height_voxel = int(height / self._voxel_size) + self.min_height_voxel
        unoccupied = np.logical_and(
            self._tsdf_vol_cpu[:, :, height_voxel] > 0, self._tsdf_vol_cpu[:, :, 0] < 0
        )  # check there is ground below

        # Set initial pose to be free
        for point in self.init_points:
            unoccupied[point[0], point[1]] = 1

        # filter small islands smaller than size 2x2 and fill in gap of size 2
        fill_size = int(fill_dim / self._voxel_size)
        structuring_element_close = np.ones((fill_size, fill_size)).astype(bool)
        unoccupied = close_operation(unoccupied, structuring_element_close)

        # Find the connected component closest to the current location is, if the current location is not free
        # this is a heuristic to determine reachable space, although not perfect
        islands = measure.label(unoccupied, connectivity=1)
        if unoccupied[cur_point[0], cur_point[1]] == 1:
            islands_ind = islands[cur_point[0], cur_point[1]]  # use current one
        else:
            # find the closest one - tbh, this should not happen, but it happens when the robot cannot see the space immediately in front of it because of camera height and fov
            y, x = np.ogrid[: unoccupied.shape[0], : unoccupied.shape[1]]
            dist_all = np.sqrt((x - cur_point[1]) ** 2 + (y - cur_point[0]) ** 2)
            dist_all[islands == islands[cur_point[0], cur_point[1]]] = np.inf
            island_coords = np.unravel_index(np.argmin(dist_all), dist_all.shape)
            islands_ind = islands[island_coords[0], island_coords[1]]
        island = islands == islands_ind
        return island, unoccupied

    def get_current_view_mask(
        self,
        cam_intr,
        cam_pose,
        im_w,
        im_h,
        slack=0,
        margin_h=0,
        margin_w=0,
    ):
        cam_pts = rigid_transform(self.cam_pts_pre, np.linalg.inv(cam_pose))
        pix_z = cam_pts[:, 2]
        pix = TSDFPlanner.cam2pix(cam_pts, cam_intr)
        pix_x, pix_y = pix[:, 0], pix[:, 1]
        valid_pix = np.logical_and(
            pix_x >= -slack + margin_w,
            np.logical_and(
                pix_x < (im_w + slack - margin_w),
                np.logical_and(
                    pix_y >= -slack + margin_h,
                    np.logical_and(pix_y < im_h + slack, pix_z > 0),
                ),
            ),
        )
        # make a 2D mask where valid pix is 1 and 0 otherwise
        valid_pix = valid_pix.reshape(self._vol_dim).astype(int)
        mask = np.max(valid_pix, axis=2)  # take the max over height (z)
        return mask

    def check_occupied_between(self, p1, p2, occupied, threshold):
        direction = np.array([p2[0] - p1[0], p2[1] - p1[1]]).astype(float)
        num_points = int(np.linalg.norm(direction))
        dir_norm = direction / np.linalg.norm(direction)
        points_between = (
            p1[:2] + dir_norm * np.arange(num_points + 1)[:, np.newaxis]
        ).astype(int)
        points_occupied = np.sum(occupied[points_between[:, 0], points_between[:, 1]])
        return points_occupied > threshold

    def check_within_bnds(self, pts, slack=0):
        return not (
            pts[0] <= slack
            or pts[0] >= self._vol_dim[0] - slack
            or pts[1] <= slack
            or pts[1] >= self._vol_dim[1] - slack
        )

    def clip_2d_array(self, array):
        return array[
            (array[:, 0] >= 0)
            & (array[:, 0] < self._vol_dim[0])
            & (array[:, 1] >= 0)
            & (array[:, 1] < self._vol_dim[1])
        ]

    def find_normal_into_space(self, point, island, space, num_check=10):
        """Find the normal direction into the space"""
        normal = find_normal(
            island.astype(int), point[0], point[1]
        )  # but normal is ambiguous, so need to find which direction is unoccupied
        dir_1 = (point + np.arange(num_check)[:, np.newaxis] * normal).astype(int)
        dir_2 = (point - np.arange(num_check)[:, np.newaxis] * normal).astype(int)
        dir_1 = self.clip_2d_array(dir_1)
        dir_2 = self.clip_2d_array(dir_2)
        dir_1_occupied = np.sum(space[dir_1[:, 0], dir_1[:, 1]])
        dir_2_occupied = np.sum(space[dir_2[:, 0], dir_2[:, 1]])
        direction = normal
        if dir_1_occupied < dir_2_occupied:
            direction *= -1
        elif dir_1_occupied == dir_2_occupied:  # randomly choose one
            if random.random() < 0.5:
                direction *= -1
        return direction

    def get_closest_distance(self, path_points: List[np.ndarray], point: np.ndarray, normal: np.ndarray, pathfinder, height):
        # get the closest distance for each segment in the path curve
        # use pathfinder's distance instead of the euclidean distance
        dist = np.inf
        cos = None

        # calculate the pathfinder distance in advance for each point in the path to reduce redundancy
        dist_list = [
            self.get_distance(point, endpoint, height, pathfinder, input_voxel=False)[0] for endpoint in path_points
        ]

        for i in range(len(path_points) - 1):
            p1, p2 = path_points[i], path_points[i + 1]
            seg = p2 - p1
            # if the point is between the two points
            if np.dot(point - p1, seg) * np.dot(point - p2, seg) <= 0:
                # get the projection of point onto the line
                t = np.dot(point - p1, seg) / np.dot(seg, seg)
                proj_point = p1 + t * seg
                d = self.get_distance(point, proj_point, height, pathfinder, input_voxel=False)[0]
            # else, get the distance to the closest endpoint
            else:
                d = min(dist_list[i], dist_list[i + 1])

            # if the distance is smaller for current edge, update
            if d < dist:
                dist = d
                cos = np.dot(seg, normal) / (np.linalg.norm(seg) * np.linalg.norm(normal))
            # if the distance is the same, update the cos value if the angle is smaller
            # this usually happens when two connected lines share the same nearest endpoint of that point
            if d == dist and np.dot(seg, normal) / (np.linalg.norm(seg) * np.linalg.norm(normal)) < cos:
                cos = np.dot(seg, normal) / (np.linalg.norm(seg) * np.linalg.norm(normal))

        return dist, cos

    @staticmethod
    def update_path_points(path_points: List[np.ndarray], point: np.ndarray):
        # get the closest line segment
        dist = np.inf
        min_dist_idx = -1
        for i in range(len(path_points) - 1):
            p1, p2 = path_points[i], path_points[i + 1]
            seg = p2 - p1
            # if the point is between the two points
            if np.dot(point - p1, seg) * np.dot(point - p2, seg) <= 0:
                d = np.abs(np.cross(seg, point - p1) / np.linalg.norm(seg))
            # else, get the distance to the closest endpoint
            else:
                d = min(np.linalg.norm(point - p1), np.linalg.norm(point - p2))
            if d < dist + 1e-6:
                dist = d
                min_dist_idx = i

        updated_path_points = path_points.copy()
        updated_path_points = updated_path_points[min_dist_idx:]

        # cut the line if point is between the two endpoints of the nearest segment
        p1, p2 = updated_path_points[0], updated_path_points[1]
        seg = p2 - p1
        if np.dot(point - p1, seg) * np.dot(point - p2, seg) <= 0:
            # find the point on segment that is closest to the point
            t = np.dot(point - p1, seg) / np.dot(seg, seg)
            closest_point = p1 + t * seg
            updated_path_points[0] = closest_point

        return updated_path_points

    @staticmethod
    def rad2vector(angle):
        return np.array([-np.sin(angle), np.cos(angle)])

    def get_distance(self, p1, p2, height, pathfinder, input_voxel=True):
        # p1, p2 are in voxel space
        # convert p1, p2 to habitat space
        if input_voxel:
            p1_world = p1 * self._voxel_size + self._vol_origin[:2]
            p2_world = p2 * self._voxel_size + self._vol_origin[:2]
        else:
            p1_world = p1
            p2_world = p2

        p1_world = np.append(p1_world, height)
        p1_habitat = pos_normal_to_habitat(p1_world)

        p2_world = np.append(p2_world, height)
        p2_habitat = pos_normal_to_habitat(p2_world)

        path = habitat_sim.ShortestPath()
        path.requested_start = p1_habitat
        path.requested_end = p2_habitat
        found_path = pathfinder.find_path(path)

        if found_path:
            return path.geodesic_distance, path.points
        else:
            if input_voxel:
                return np.linalg.norm(p1 - p2) * self._voxel_size, None
            else:
                return np.linalg.norm(p1 - p2), None

    def habitat2voxel(self, pts):
        pts_normal = pos_habitat_to_normal(pts)
        pts_voxel = self.world2vox(pts_normal)
        return pts_voxel

    def get_frontier_region_map(self, frontier_coordinates):
        # frontier_coordinates: [N, 2] ndarray of the coordinates covered by the frontier in voxel space
        region_map = np.zeros_like(self.frontier_map, dtype=bool)
        for coord in frontier_coordinates:
            region_map[coord[0], coord[1]] = True
        return region_map

    def create_frontier(self, ft_data: dict, frontier_edge_areas, cur_point) -> Frontier:
        ft_direction = np.array([np.cos(ft_data['angle']), np.sin(ft_data['angle'])])

        kernel = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ])
        frontier_edge = ndimage.convolve(ft_data['region'].astype(int), kernel, mode='constant', cval=0)

        frontier_edge_areas_filtered = np.asarray(
            [p for p in frontier_edge_areas if 2 <= frontier_edge[p[0], p[1]] <= 12]
        )
        if len(frontier_edge_areas_filtered) > 0:
            frontier_edge_areas = frontier_edge_areas_filtered

        all_directions = frontier_edge_areas - cur_point[:2]
        all_direction_norm = np.linalg.norm(all_directions, axis=1, keepdims=True)
        all_direction_norm = np.where(all_direction_norm == 0, np.inf, all_direction_norm)
        all_directions = all_directions / all_direction_norm

        # the center is the closest point in the edge areas from current point that have close cosine angles
        cos_sim_rank = np.argsort(-np.dot(all_directions, ft_direction))
        center_candidates = np.asarray(
            [frontier_edge_areas[idx] for idx in cos_sim_rank[:5]]
        )
        center = center_candidates[
            np.argmin(np.linalg.norm(center_candidates - cur_point[:2], axis=1))
        ]
        center = adjust_navigation_point(
            center, self.occupied, max_dist=0.5, max_adjust_distance=0.3, voxel_size=self._voxel_size
        )

        # center = frontier_edge_areas[
        #     np.argmax(np.dot(all_directions, ft_direction))
        # ]

        # cos_sim_rank = np.argsort(-np.dot(all_directions, ft_direction))
        # # the center is the farthest point in the closest three points
        # center_candidates = np.asarray(
        #     [frontier_edge_areas[idx] for idx in cos_sim_rank[:6]]
        # )
        # center = center_candidates[
        #     np.argmax(np.linalg.norm(center_candidates - cur_point[:2], axis=1))
        # ]

        region = ft_data['region']

        # allocate an id for the frontier
        # assert np.all(self.frontier_map[region] == 0)
        frontier_id = self.frontier_counter
        self.frontier_map[region] = frontier_id
        self.frontier_counter += 1

        return Frontier(
            position=center,
            orientation=ft_direction,
            region=region,
            frontier_id=frontier_id
        )

    def free_frontier(self, frontier: Frontier):
        self.frontier_map[self.frontier_map == frontier.frontier_id] = 0


