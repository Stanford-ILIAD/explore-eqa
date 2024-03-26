import numpy as np
import random
import scipy.ndimage as ndimage
import heapq
import math
from src.habitat import pos_habitat_to_normal


def get_scene_bnds(pathfinder, floor_height):
    # Get mesh boundaries - this is for the full scene
    scene_bnds = pathfinder.get_bounds()
    scene_lower_bnds_normal = pos_habitat_to_normal(scene_bnds[0])
    scene_upper_bnds_normal = pos_habitat_to_normal(scene_bnds[1])
    scene_size = np.abs(
        np.prod(scene_upper_bnds_normal[:2] - scene_lower_bnds_normal[:2])
    )
    tsdf_bnds = np.array(
        [
            [
                min(scene_lower_bnds_normal[0], scene_upper_bnds_normal[0]),
                max(scene_lower_bnds_normal[0], scene_upper_bnds_normal[0]),
            ],
            [
                min(scene_lower_bnds_normal[1], scene_upper_bnds_normal[1]),
                max(scene_lower_bnds_normal[1], scene_upper_bnds_normal[1]),
            ],
            [
                floor_height - 0.2,
                floor_height + 3.5,
            ],
        ]
    )
    return tsdf_bnds, scene_size


def get_cam_intr(hfov, img_height, img_width):
    hfov_rad = hfov * np.pi / 180
    vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) * img_height / img_width)
    # vfov = vfov_rad * 180 / np.pi
    fx = (1.0 / np.tan(hfov_rad / 2.0)) * img_width / 2.0
    fy = (1.0 / np.tan(vfov_rad / 2.0)) * img_height / 2.0
    cx = img_width // 2
    cy = img_height // 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def fps(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = np.array(points)

    # Represent the points by their indices in points
    points_left = np.arange(len(points))  # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype="int")  # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float("inf")  # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected)  # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]

        dist_to_last_added_point = (
            (points[last_added] - points[points_left]) ** 2
        ).sum(
            -1
        )  # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(
            dist_to_last_added_point, dists[points_left]
        )  # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return points[sample_inds]


def points_in_circle(center_x, center_y, radius, grid_shape):
    x, y = np.meshgrid(
        np.arange(grid_shape[0]), np.arange(grid_shape[1]), indexing="ij"
    )  # use matrix indexing instead of cartesian
    distance_matrix = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    points_within_circle = np.where(distance_matrix <= radius)
    return list(zip(points_within_circle[0], points_within_circle[1]))


def run_dijkstra(grid, start, end):
    start = tuple(start[:2])
    end = tuple(end[:2])
    rows, cols = grid.shape
    directions = [
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ]  # 8 directions
    distance = np.full(grid.shape, np.inf)
    distance[start] = 0
    prev = {start: None}

    # Priority queue for the nodes to explore
    pq = [(0, start)]  # (distance, node)

    while pq:
        dist, current = heapq.heappop(pq)

        if current == end:
            break

        for direction in directions:
            r, c = current[0] + direction[0], current[1] + direction[1]
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0:
                new_dist = dist + math.sqrt(direction[0] ** 2 + direction[1] ** 2)
                if new_dist < distance[r, c]:
                    distance[r, c] = new_dist
                    heapq.heappush(pq, (new_dist, (r, c)))
                    prev[(r, c)] = current

    # Reconstructing path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = prev.get(current)
    return path[::-1]  # Return reversed path


def find_normal(grid, x, y):
    # Sobel operators
    sobel_y = ndimage.sobel(grid, axis=0)  # gradient on y
    sobel_x = ndimage.sobel(grid, axis=1)  # gradient on x

    # Gradient at (x,y)
    Gx = sobel_x[x, y]
    Gy = sobel_y[x, y]

    # # If the gradient is near zero, we might be in a flat region or not on an edge
    if Gx == 0 and Gy == 0:
        # random
        normal = np.array([random.random(), random.random()])
    else:
        # Normal of the edge is perpendicular to the gradient
        normal = np.array([-Gy, -Gx])  # need to negate x???

    # Normalize the vector (make its length 1)
    normal = normal / np.linalg.norm(normal)
    return normal


def open_operation(array, structure=None):
    eroded = ndimage.binary_erosion(array, structure=structure)
    opened = ndimage.binary_dilation(eroded, structure=structure)
    return opened


def close_operation(array, structure=None):
    dilated = ndimage.binary_dilation(array, structure=structure)
    closed = ndimage.binary_erosion(dilated, structure=structure)
    return closed


def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud."""
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image"""
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array(
        [
            (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2])
            * np.array([0, max_depth, max_depth, max_depth, max_depth])
            / cam_intr[0, 0],
            (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2])
            * np.array([0, max_depth, max_depth, max_depth, max_depth])
            / cam_intr[1, 1],
            np.array([0, max_depth, max_depth, max_depth, max_depth]),
        ]
    )
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts


def meshwrite(filename, verts, faces, norms, colors):
    """Save a 3D mesh to a polygon .ply file."""
    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write(
            "%f %f %f %f %f %f %d %d %d\n"
            % (
                verts[i, 0],
                verts[i, 1],
                verts[i, 2],
                norms[i, 0],
                norms[i, 1],
                norms[i, 2],
                colors[i, 0],
                colors[i, 1],
                colors[i, 2],
            )
        )

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file."""
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write(
            "%f %f %f %d %d %d\n"
            % (
                xyz[i, 0],
                xyz[i, 1],
                xyz[i, 2],
                rgb[i, 0],
                rgb[i, 1],
                rgb[i, 2],
            )
        )
