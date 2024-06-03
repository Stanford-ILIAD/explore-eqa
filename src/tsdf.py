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
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
from .geom import (
    points_in_circle,
    find_normal,
    close_operation,
    rigid_transform,
    run_dijkstra,
    fps,
    get_nearest_true_point,
    get_proper_observe_point
)
from .habitat import pos_normal_to_habitat, pos_habitat_to_normal
import habitat_sim
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Frontier:
    """Frontier class for frontier-based exploration."""

    position: np.ndarray  # integer position in voxel grid
    orientation: np.ndarray  # directional vector of the frontier in float
    image: Optional[str]
    area: int
    visited: bool = False  # whether the frontier has been visited before

    def __eq__(self, other):
        if not isinstance(other, Frontier):
            raise TypeError("Cannot compare Frontier with non-Frontier object.")
        # two frontiers are equal if they have the same position, and their orientations are close
        return (np.array_equal(self.position, other.position) and
                np.dot(self.orientation, other.orientation) / (np.linalg.norm(self.orientation) * np.linalg.norm(other.orientation)) > 0.9)


@dataclass
class Object:
    """Object class for semantic objects."""

    position: np.ndarray  # integer position in voxel grid


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

        self.target_point = None
        self.target_direction = None
        self.max_point = None
        self.simple_scene_graph = {}
        self.frontiers: List[Frontier] = []

    def increment_scene_graph(self, semantic_obs, obj_id_to_bbox, min_pix_ratio=0.0):
        unique_semantic_ids = np.unique(semantic_obs)
        for obj_id in unique_semantic_ids:
            if obj_id not in self.simple_scene_graph.keys() and obj_id != 0 and obj_id in obj_id_to_bbox.keys():
                bbox_data = obj_id_to_bbox[obj_id]
                if bbox_data['class'] in ["wall", "floor", "ceiling"]:
                    continue
                if np.sum(semantic_obs == obj_id) / (semantic_obs.shape[0] * semantic_obs.shape[1]) < min_pix_ratio:
                    continue
                bbox = bbox_data["bbox"]
                bbox = np.asarray(bbox)
                bbox_center = np.mean(bbox, axis=0)
                # change to x, z, y for habitat
                bbox_center = bbox_center[[0, 2, 1]]
                # add to simple scene graph
                self.simple_scene_graph[obj_id] = bbox_center

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

    def find_prompt_points_within_view(
        self,
        pts,
        im_w,
        im_h,
        cam_intr,
        cam_pose,
        height=0.4,
        cluster_threshold=1.0,
        num_prompt_points=3,
        num_max_unoccupied=300,
        min_points_for_clustering=3,
        point_min_dist=2,
        point_max_dist=10,
        cam_offset=0.5,
        **kwargs,
    ):
        """Find locations within view, which will then be prompted with VLM to get their semantic values.
        Locations include:
            (1) frontiers within view
            (2) empty locations that are sufficiently far from the current point within view
        """
        cur_point = self.world2vox(pts)
        island, unoccupied = self.get_island_around_pts(pts, height=height)
        unexplored = (np.sum(self._explore_vol_cpu, axis=-1) == 0).astype(int)
        for point in self.init_points:
            unexplored[point[0], point[1]] = 0
        occupied = np.logical_not(unoccupied).astype(int)
        cam_pose = cam_pose @ np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, cam_offset],
                [0, 0, 0, 1],
            ]
        )
        mask = self.get_current_view_mask(
            cam_intr, cam_pose, im_w, im_h, slack=0, margin_h=100, margin_w=30
        )

        ############## Get unoccupied reachable points in view ##############

        # Mask the unoccupied region to be only the current view
        unoccupied_in_view = np.multiply(unoccupied, mask)
        unoccupied_reachable_in_view = np.argwhere((island) & (unoccupied_in_view))

        # Subsample - weigh closer points more
        if len(unoccupied_reachable_in_view) > 0:
            subsample_inds = np.random.choice(
                range(len(unoccupied_reachable_in_view)),
                min(num_max_unoccupied, len(unoccupied_reachable_in_view)),
                replace=False,
            )
            unoccupied_reachable_in_view = unoccupied_reachable_in_view[subsample_inds]
        else:
            unoccupied_reachable_in_view = np.empty((0, 2))

        # Check unoccupied between point and current point - skip if any occupied
        unoccupied_reachable_in_view_new = np.empty((0, 2))
        for point in unoccupied_reachable_in_view:
            if not self.check_occupied_between(point, cur_point, occupied, threshold=1):
                unoccupied_reachable_in_view_new = np.concatenate(
                    (unoccupied_reachable_in_view_new, [point]), axis=0
                )
        unoccupied_reachable_in_view = unoccupied_reachable_in_view_new

        # Only keep points within desired range
        if len(unoccupied_reachable_in_view) > 0:
            dist_all = np.sqrt(
                (unoccupied_reachable_in_view[:, 0] - cur_point[0]) ** 2
                + (unoccupied_reachable_in_view[:, 1] - cur_point[1]) ** 2
            )
            unoccupied_reachable_in_view = unoccupied_reachable_in_view[
                (dist_all > point_min_dist / self._voxel_size)
                & (dist_all < point_max_dist / self._voxel_size)
            ]
            dist_all = dist_all[
                (dist_all > point_min_dist / self._voxel_size)
                & (dist_all < point_max_dist / self._voxel_size)
            ]

        ################## Get frontiers in view ##################

        # Get unexplored region - mark points around init points to be explored
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        unexplored_neighbors = ndimage.convolve(
            unexplored, kernel, mode="constant", cval=0.0
        )
        frontiers_in_view = np.empty((0, 2))

        ################## Combine points, cluster ##################
        candidates_pre_cluster = np.concatenate(
            [frontiers_in_view, unoccupied_reachable_in_view], axis=0
        )

        # initialize plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
        ax1.imshow(unoccupied)
        ax1.scatter(cur_point[1], cur_point[0], c="b", s=30)
        for point in candidates_pre_cluster:
            ax1.scatter(point[1], point[0], c="r", s=8, alpha=0.5)
        ax1.set_title("Unoccupied")
        ax2.imshow(unexplored)
        ax2.scatter(cur_point[1], cur_point[0], c="b", s=30)
        for point in frontiers_in_view:
            ax2.scatter(point[1], point[0], c="r", s=8, alpha=0.5)
        ax2.set_title("Unexplored")
        ax3.imshow(island)
        ax3.set_title("Island")

        # cluster, or return none
        if len(candidates_pre_cluster) < min_points_for_clustering:
            candidates_pix = np.empty((0, 2))
        else:
            clusters = fps(candidates_pre_cluster, num_prompt_points)

            # merge clusters if too close to each other
            clusters_new = np.empty((0, 2))
            for cluster in clusters:
                if len(clusters_new) == 0:
                    clusters_new = np.vstack((clusters_new, cluster))
                else:
                    clusters_array = np.array(clusters_new)
                    dist = np.sqrt(np.sum((clusters_array - cluster) ** 2, axis=1))
                    if np.min(dist) > cluster_threshold / self._voxel_size:
                        clusters_new = np.vstack((clusters_new, cluster))
            candidates = clusters_new
            self.candidates = candidates
            logging.info(f"Number of final candidates: {len(candidates)}")

            # add final points to plots
            for ax in [ax1, ax2]:
                for point in candidates:
                    ax.scatter(point[1], point[0], c="g", s=30)
                for point in self.simple_scene_graph.values():
                    vox_point = self.world2vox(np.array(point))
                    ax.scatter(vox_point[1], vox_point[0], c="w", s=30)

            # Convert to pixel coordinates
            if len(candidates) > 0:
                candidates_cam = [
                    rigid_transform(
                        TSDFPlanner.vox2world(
                            self._vol_origin,
                            np.append(candidates[i], 0).reshape(1, 3),
                            self._voxel_size,
                        ),
                        np.linalg.inv(cam_pose),
                    )
                    for i in range(len(candidates))
                ]  # to camera coordinates first
                candidates_cam = np.concatenate(candidates_cam, axis=0)
                candidates_pix = TSDFPlanner.cam2pix(candidates_cam, cam_intr)
            else:
                candidates_pix = np.empty((0, 2))

        # Save global info
        self.cur_point, self.island, self.unexplored = cur_point, island, unexplored
        self.unoccupied, self.occupied = unoccupied, occupied
        self.unexplored_neighbors = unexplored_neighbors

        return candidates_pix, fig

    def find_next_pose_with_path(
        self,
        pts,
        angle,
        path_points,
        pathfinder,
        target_obj_id,
        flag_no_val_weight=False,
        unexplored_T=0.5,
        unoccupied_T=3,
        val_T=0.5,
        val_dir_T=0.5,
        dist_T=10,
        min_dist_from_cur=0.5,
        max_dist_from_cur_phase_1=3,
        max_dist_from_cur_phase_2=1,
        frontier_spacing=1.5,
        frontier_min_neighbors=3,
        frontier_max_neighbors=4,
        max_unexplored_check_frontier=3.0,
        max_unoccupied_check_frontier=1.0,
        max_val_check_frontier=5.0,
        smooth_sigma=5,
        eps=0.5,
        min_frontier_area=6,
        final_observe_distance=1.0,
        **kwargs,
    ):
        """Determine the next frontier to traverse to with semantic-value-weighted sampling."""
        cur_point = self.world2vox(pts)
        if hasattr(self, "cur_point"):
            island = self.island
            unoccupied, occupied = self.unoccupied, self.occupied
            unexplored, unexplored_neighbors = (
                self.unexplored,
                self.unexplored_neighbors,
            )
        else:
            island, unoccupied = self.get_island_around_pts(pts, height=0.4)
            occupied = np.logical_not(unoccupied).astype(int)
            unexplored = (np.sum(self._explore_vol_cpu, axis=-1) == 0).astype(int)
            for point in self.init_points:
                unexplored[point[0], point[1]] = 0
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
            unexplored_neighbors = ndimage.convolve(
                unexplored, kernel, mode="constant", cval=0.0
            )
        self.unexplored_neighbors = unexplored_neighbors
        self.unoccupied = unoccupied

        # get semantic map by taking max over z
        val_vol_2d = np.max(self._val_vol_cpu, axis=2).copy()

        # smoothen the map
        val_vol_2d = gaussian_filter(val_vol_2d, sigma=smooth_sigma)

        # detect and update frontiers
        frontiers_regions = np.argwhere(
            island
            & (unexplored_neighbors >= frontier_min_neighbors)
            & (unexplored_neighbors <= frontier_max_neighbors)
        )
        frontiers_pre_cluster = frontiers_regions.copy()

        # cluster frontier regions
        db = DBSCAN(eps=eps, min_samples=2).fit(frontiers_regions)
        labels = db.labels_
        # get one point from each cluster
        frontier_list = []
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster = frontiers_regions[labels == label]
            area = len(cluster)

            # take the one that is closest to mean as the center of the cluster
            dist = np.sqrt(
                (cluster[:, 0] - np.mean(cluster[:, 0])) ** 2
                + (cluster[:, 1] - np.mean(cluster[:, 1])) ** 2
            )
            min_dist_rank = np.argsort(dist)
            center = None
            while True:
                if len(min_dist_rank) == 0:
                    break
                center = cluster[min_dist_rank[0]]
                if island[center[0], center[1]] and self.check_within_bnds(center):  # ensure the center is within the island
                    break
                min_dist_rank = min_dist_rank[1:]
            if center is None:
                continue

            if np.linalg.norm(center - cur_point[:2]) < 1e-3:
                # skip the frontier if it is too near to the agent
                continue

            # the orientation of the frontier the average of the vector from agent to each point in the cluster
            ft_direction = np.mean(cluster - cur_point[:2], axis=0)
            ft_direction = ft_direction / np.linalg.norm(ft_direction)

            frontier_list.append(
                Frontier(center, ft_direction, None, area)
            )

        # remove frontiers that did not appear in current observation
        self.frontiers = [frontier for frontier in self.frontiers if any(frontier == curr_ft for curr_ft in frontier_list)]
        # add new frontiers in this observation
        for frontier in frontier_list:
            if not any(frontier == prev_ft for prev_ft in self.frontiers):
                self.frontiers.append(frontier)

        # subsample
        frontiers_weight = np.zeros((len(self.frontiers)))

        # determine whether the target object is in scene graph
        if target_obj_id in self.simple_scene_graph.keys():
            target_point = self.habitat2voxel(self.simple_scene_graph[target_obj_id])[:2]
            # # set the object center as the navigation target
            # target_navigable_point = get_nearest_true_point(target_point, unoccupied)  # get the nearest unoccupied point for the nav target
            # since it's not proper to directly go to the target point,
            # we'd better find a navigable point that is certain distance from it to better observe the target
            target_navigable_point = get_proper_observe_point(target_point, unoccupied, dist=final_observe_distance / self._voxel_size)
            if target_navigable_point is None:
                # a wierd case that no unoccupied point is found in all the space
                raise ValueError("No unoccupied point is found in the scene")
            self.target_point = target_navigable_point
            self.max_point = Object(target_point.astype(int))
            logging.info(f"Target point found: {target_point}")

        # if target point is not found, find the next frontier
        if self.target_point is None:
            point_type = "current"
            # Get weights for frontiers
            frontiers_weight = np.empty(0)
            for frontier in self.frontiers:
                # find normal of the frontier
                normal = frontier.orientation

                # Then check how much unoccupied in that direction
                max_pixel_check = int(max_unoccupied_check_frontier / self._voxel_size)
                dir_pts = np.round(
                    frontier.position + np.arange(max_pixel_check)[:, np.newaxis] * normal
                ).astype(int)
                dir_pts = self.clip_2d_array(dir_pts)
                unoccupied_rate = (
                    np.sum(unoccupied[dir_pts[:, 0], dir_pts[:, 1]] == 1)
                    / max_pixel_check
                )

                # Check the ratio of unexplored in the direction, until hits obstacle
                max_pixel_check = int(max_unexplored_check_frontier / self._voxel_size)
                dir_pts = np.round(
                    frontier.position + np.arange(max_pixel_check)[:, np.newaxis] * normal
                ).astype(int)
                dir_pts = self.clip_2d_array(dir_pts)
                unexplored_rate = (
                    np.sum(unexplored[dir_pts[:, 0], dir_pts[:, 1]] == 1)
                    / max_pixel_check
                )

                # Check value in the direction
                # max_pixel_check = int(max_val_check_frontier / self._voxel_size)
                # dir_pts = np.round(
                #     point + np.arange(max_pixel_check)[:, np.newaxis] * normal
                # ).astype(int)
                # dir_pts = self.clip_2d_array(dir_pts)
                # val_vol_2d_dir = val_vol_2d[dir_pts[:, 0], dir_pts[:, 1]]
                # # keep non zero value only
                # val_vol_2d_dir = val_vol_2d_dir[val_vol_2d_dir > 0]
                # if len(val_vol_2d_dir) == 0:
                #     val = 0
                # else:
                #     val = np.mean(val_vol_2d_dir)

                # get weight for path points
                pos_world = frontier.position * self._voxel_size + self._vol_origin[:2]
                closest_dist, cosine_dist = self.get_closest_distance(path_points, pos_world, normal, pathfinder, pts[2])

                # Get weight - unexplored, unoccupied, and value
                weight = np.exp(unexplored_rate / unexplored_T)  # [0-1] before T
                weight *= np.exp(unoccupied_rate / unoccupied_T)  # [0-1] before T
                # if not flag_no_val_weight:
                #     weight *= np.exp(
                #         val_vol_2d[point[0], point[1]] / val_T
                #     )  # [0-1] before T
                #     weight *= np.exp(val / val_dir_T)  # [0-1] before T

                # add weight for path points
                weight *= np.exp(- closest_dist) * 3
                weight *= np.exp(cosine_dist)

                # Check distance to current point - make weight very small if too close and aligned
                dist = np.sqrt((cur_point[0] - frontier.position[0]) ** 2 + (cur_point[1] - frontier.position[1]) ** 2)
                pts_angle = np.arctan2(normal[1], normal[0]) - np.pi / 2
                weight *= np.exp(-dist / dist_T)
                if (
                    dist < min_dist_from_cur / self._voxel_size
                    and np.abs(angle - pts_angle) < np.pi / 6
                ):
                    weight *= 1e-3

                # if frontier is too small, ignore it
                if frontier.area < min_frontier_area:
                    weight *= 1e-3

                # if the frontier is visited before, ignore it
                if frontier.visited:
                    weight *= 1e-3

                # Save weight
                frontiers_weight = np.append(frontiers_weight, weight)
            logging.info(f"Number of frontiers for next pose: {len(self.frontiers)}")

            # choose the frontier with highest weight
            if len(self.frontiers) > 0:
                logging.info(
                    f"Mean and std of frontier weight: {np.mean(frontiers_weight):.3f},"
                    f" {np.std(frontiers_weight):.3f}"
                )
                point_type = "frontier"

                # take best point until it satisfies condition
                max_try = 50
                cnt_try = 0
                while 1:
                    cnt_try += 1
                    if cnt_try > max_try:
                        point_type = "current"
                        break
                    frontiers_weight_red = frontiers_weight / np.mean(
                        frontiers_weight
                    )  # prevent overflowing
                    # change from random choose to choose the max weight
                    # frontier_ind = np.random.choice(
                    #     range(len(frontiers)),
                    #     p=frontiers_weight_red / np.sum(frontiers_weight_red),
                    # )
                    frontier_ind = np.argmax(frontiers_weight)
                    logging.info(f"weight: {frontiers_weight[frontier_ind]:.3f}")
                    max_point = self.frontiers[frontier_ind]

                    # find the direction into unexplored
                    ft_direction = max_point.orientation

                    # The following code add backtrack to the frontier
                    # # Move back in the opposite direction of the normal by spacing, so the robot can see the frontier
                    # # there is a chance that the point is outside the free space
                    # next_point = np.array(max_point.position, dtype=float)
                    # max_backtrack = int(frontier_spacing / self._voxel_size)
                    # num_backtrack = 0
                    # best_clearance_backtrack_point = None
                    # best_clearance = 0
                    # clearance = 0
                    # # traverse all the backtrack points, and find the valid point with the largest clearance ahead
                    # while num_backtrack < max_backtrack:
                    #     next_point -= ft_direction
                    #     num_backtrack += 1
                    #
                    #     # if the point is invalid
                    #     if (occupied[int(next_point[0]), int(next_point[1])]
                    #         or not island[int(np.round(next_point[0])), int(np.round(next_point[1]))]
                    #         or not self.check_within_bnds(next_point)  # out of bound of the map
                    #     ):
                    #         clearance = 0
                    #         continue
                    #     else:
                    #         clearance += 1
                    #         if clearance > best_clearance:
                    #             best_clearance = clearance
                    #             best_clearance_backtrack_point = next_point.copy()
                    # if best_clearance_backtrack_point is None:
                    #     # all points backward are invalid
                    #     # so just skip this frontier
                    #     logging.info("All points backward are invalid, skip this frontier!!!!!")
                    #     continue

                    # next_point = np.round(best_clearance_backtrack_point).astype(int)

                    # this try not backtrack
                    next_point = np.array(max_point.position, dtype=float)
                    while (
                        occupied[int(np.round(next_point[0])), int(np.round(next_point[1]))] or
                        not island[int(np.round(next_point[0])), int(np.round(next_point[1]))] or
                        not self.check_within_bnds(next_point)
                    ):
                        next_point -= ft_direction

                    next_point = np.round(next_point).astype(int)
                    if (
                        self.check_within_bnds(next_point)
                        and island[next_point[0], next_point[1]]
                    ):
                        break  # stop searching

            # no patch used
            if point_type == "current":
                logging.info("No patches, return current point and random direction")
                next_point = cur_point[:2]
                max_point = Object(next_point.astype(int))
        else:  # target point is found, then go directly to the target point
            point_type = "commit"
            next_point = self.target_point.copy()
            max_point = copy.deepcopy(self.max_point)
        logging.info(f"Next pose type: {point_type}")

        # check the distance to next navigation point
        # if the target navigation point is too far
        # then just go to a point between the current point and the target point
        max_dist_from_cur = max_dist_from_cur_phase_1 if self.target_point is None else max_dist_from_cur_phase_2  # in phase 2, the step size should be smaller
        dist, path_to_target = self.get_distance(cur_point[:2], next_point, height=pts[2], pathfinder=pathfinder)
        # drop the y value of the path to avoid errors when calculating seg_length
        path_to_target = [np.asarray([p[0], 0.0, p[2]]) for p in path_to_target]

        if dist > max_dist_from_cur:
            if path_to_target is not None:
                # if the pathfinder find a path, then just walk along the path for max_dist_from_cur distance
                dist_to_travel = max_dist_from_cur
                for i in range(len(path_to_target) - 1):
                    seg_length = np.linalg.norm(path_to_target[i + 1] - path_to_target[i])
                    if seg_length < dist_to_travel:
                        dist_to_travel -= seg_length
                    else:
                        # find the point on the segment according to the length ratio
                        next_point_habitat = path_to_target[i] + (path_to_target[i + 1] - path_to_target[i]) * dist_to_travel / seg_length
                        next_point = self.world2vox(pos_habitat_to_normal(next_point_habitat))[:2]
                        break
            else:
                # if the pathfinder cannot find a path, then just go to a point between the current point and the target point
                walk_dir = next_point - cur_point[:2]
                walk_dir = walk_dir / np.linalg.norm(walk_dir)
                next_point = cur_point[:2] + (next_point - cur_point[:2]) * max_dist_from_cur / dist
                # ensure next point is valid, otherwise go backward a bit
                while (
                    not self.check_within_bnds(next_point)
                    or not island[int(np.round(next_point[0])), int(np.round(next_point[1]))]
                    or occupied[int(np.round(next_point[0])), int(np.round(next_point[1]))]
                ):
                    next_point -= walk_dir * 0.3 / self._voxel_size
                next_point = np.round(next_point).astype(int)

        # determine the direction: from next point to max point
        if np.array_equal(next_point.astype(int), max_point.position):  # if the next point is the max point
            # this case should not happen actually, since the exploration should end before this
            if np.array_equal(cur_point[:2], next_point):  # if the current point is also the max point
                # then just set some random direction
                direction = np.random.randn(2)
            else:
                direction = max_point.position - cur_point[:2]
        else:
            # normal direction from next point to max point
            direction = max_point.position - next_point
        direction = direction / np.linalg.norm(direction)

        # mark the max point as visited if it is a frontier
        if type(max_point) == Frontier:
            max_point.visited = True

        # Plot
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 18))
        agent_orientation = self.rad2vector(angle)
        ax1.imshow(unoccupied)
        ax1.scatter(max_point.position[1], max_point.position[0], c="r", s=30, label="max")
        ax1.scatter(cur_point[1], cur_point[0], c="b", s=30, label="current")
        ax1.arrow(cur_point[1], cur_point[0], agent_orientation[1] * 4, agent_orientation[0] * 4, width=0.1, head_width=0.8, head_length=0.8, color='b')
        ax1.scatter(next_point[1], next_point[0], c="g", s=30, label="actual")
        # plot all the detected objects
        for obj_center in self.simple_scene_graph.values():
            obj_vox = self.habitat2voxel(obj_center)
            ax1.scatter(obj_vox[1], obj_vox[0], c="w", s=30)
        # plot the target point if found
        if self.target_point is not None:
            ax1.scatter(max_point.position[1], max_point.position[0], c="r", s=80, label="target")
        ax1.set_title("Unoccupied")

        ax2.imshow(island)
        for frontier in self.frontiers:
            if frontier.image is not None:
                ax2.scatter(frontier.position[1], frontier.position[0], color="g", s=50, alpha=1)
            else:
                ax2.scatter(frontier.position[1], frontier.position[0], color="r", s=50, alpha=1)
        ax2.scatter(cur_point[1], cur_point[0], c="b", s=30, label="current")
        ax2.set_title("Island")

        ax3.imshow(unexplored_neighbors)
        for point in frontiers_pre_cluster:
            ax3.scatter(point[1], point[0], color="white", s=20, alpha=1)
        for frontier in self.frontiers:
            ax3.scatter(frontier.position[1], frontier.position[0], color="m", s=10, alpha=1)
            normal = frontier.orientation
            dx, dy = normal * 4
            ax3.arrow(frontier.position[1], frontier.position[0], dy, dx, width=0.1, head_width=0.8, head_length=0.8, color='m')
        ax3.scatter(max_point.position[1], max_point.position[0], c="r", s=30, label="max")
        ax3.set_title("Unexplored neighbors")

        im = ax4.imshow(val_vol_2d)
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

        im = ax5.imshow(island)
        ax5.set_title("Path on island")

        frontier_weights = np.zeros_like(val_vol_2d)
        for frontier, weight in zip(self.frontiers, frontiers_weight):
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
            return np.linalg.norm(p1 - p2), None

    def habitat2voxel(self, pts):
        pts_normal = pos_habitat_to_normal(pts)
        pts_voxel = self.world2vox(pts_normal)
        return pts_voxel












