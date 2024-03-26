import numpy as np
import habitat_sim


def pos_normal_to_habitat(pts):
    # +90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))


def pos_habitat_to_normal(pts):
    # -90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))


def pose_habitat_to_normal(pose):
    # T_normal_cam = T_normal_habitat * T_habitat_cam
    return np.dot(
        np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), pose
    )


def pose_normal_to_tsdf(pose):
    return np.dot(
        pose, np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    )


def pose_normal_to_tsdf_real(pose):
    # This one makes sense, which is making x-forward, y-left, z-up to z-forward, x-right, y-down
    return pose @ np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])


def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor, a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.hfov = settings["hfov"]

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.hfov = settings["hfov"]

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def merge_pointcloud(
    pts_orig, pts_new, clip_grads=None, new_clip_grads=None, threshold=1e-2
):
    """Merge two pointclouds, do not add if already exists. Add clip grads if provided.
    Args:
        pts_orig: Nx3 float array of 3D points
        pts_new: Mx3 float array of 3D points
        clip_grads: NxK float array of clip grads
        new_clip_grads: MxK float array of clip grads
    Returns:
        pts_orig: Nx3 float array of merged 3D points
        clip_grads: NxK float array of merged clip grads
    """
    pts_orig = np.vstack((pts_orig, pts_new))
    # merge points that are too close
    close_point_sets = []
    visited = np.zeros(len(pts_orig), dtype=bool)
    for i in range(len(pts_orig)):
        if not visited[i]:
            close_points = np.linalg.norm(pts_orig - pts_orig[i], axis=1) < threshold
            visited[close_points] = True
            close_point_sets.append(np.where(close_points)[0].tolist())

    # get new point cloud
    pts_orig = np.array(
        [np.mean(pts_orig[point_set], axis=0) for point_set in close_point_sets]
    )

    # add clip grads, also take average
    if clip_grads is not None:
        clip_grads = np.vstack((clip_grads, new_clip_grads))
        clip_grads = np.array(
            [np.mean(clip_grads[point_set], axis=0) for point_set in close_point_sets]
        )
        return pts_orig, clip_grads
    return pts_orig


def transform_pointcloud(xyz_pts, rigid_transform):
    """Apply rigid transformation to 3D pointcloud.
    Args:
        xyz_pts: Nx3 float array of 3D points
        rigid_transform: 3x4 or 4x4 float array defining a rigid transformation (rotation and translation)
    Returns:
        xyz_pts: Nx3 float array of transformed 3D points
    """
    xyz_pts = np.dot(rigid_transform[:3, :3], xyz_pts.T)  # apply rotation
    xyz_pts = xyz_pts + np.tile(
        rigid_transform[:3, 3].reshape(3, 1), (1, xyz_pts.shape[1])
    )  # apply translation
    return xyz_pts.T


def get_pointcloud(depth, hfov, cam_pose=None):
    """Get 3D pointcloud from depth image. Calculate camera intrinsics based on image sizes and hfov."""
    H, W = depth.shape
    hfov = hfov * np.pi / 180  # deg2rad
    # vfov = 2 * np.arctan(np.tan(hfov / 2) * H / W)
    # fx = (1.0 / np.tan(hfov / 2.)) * W / 2.0
    # fy = (1.0 / np.tan(vfov / 2.)) * H / 2.0
    # cx = W // 2
    # cy = H // 2

    # # Project depth into 3D pointcloud in camera coordinates
    # pixel_x, pixel_y = np.meshgrid(np.linspace(0, img_w - 1, img_w),
    #                                np.linspace(0, img_h - 1, img_h))
    # cam_pts_x = ((pixel_x - cx) / fx) * depth
    # cam_pts_y = ((pixel_y - cy) / fy) * depth
    # cam_pts_z = -depth
    # cam_pts = (np.array([cam_pts_x, cam_pts_y,
    #                      cam_pts_z]).transpose(1, 2, 0).reshape(-1, 3))

    K = np.array(
        [
            [1 / np.tan(hfov / 2.0), 0.0, 0.0, 0.0],
            [0.0, 1 / np.tan(hfov / 2.0) * W / H, 0.0, 0.0],
            [0.0, 0.0, 1, 0],
            [0.0, 0.0, 0, 1],
        ]
    )

    # Now get an approximation for the true world coordinates -- see if they make sense
    # [-1, 1] for x and [1, -1] for y as array indexing is y-down while world is y-up
    xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))
    depth = depth.reshape(1, W, H)
    xs = xs.reshape(1, W, H)
    ys = ys.reshape(1, W, H)

    # Unproject
    # negate depth as the camera looks along -Z
    xys = np.vstack((xs * depth, ys * depth, -depth, np.ones(depth.shape)))
    xys = xys.reshape(4, -1)
    cam_pts = np.matmul(np.linalg.inv(K), xys)
    cam_pts = cam_pts.T[:, :3]

    # # Transform to world coordinates
    if cam_pose is not None:
        # cam_pts = transform_pointcloud(cam_pts, np.linalg.inv(cam_pose))
        cam_pts = transform_pointcloud(cam_pts, cam_pose)

    # Flip axes?
    cam_pts = np.hstack((cam_pts[:, 0:1], -cam_pts[:, 2:3], cam_pts[:, 1:2]))
    # print(np.min(cam_pts, axis=0), np.max(cam_pts, axis=0))
    return cam_pts


def rgba2rgb(rgba, background=(1, 1, 1)):
    row, col, ch = rgba.shape
    if ch == 3:
        return rgba
    assert ch == 4, "RGBA image has 4 channels."
    rgb = np.zeros((row, col, 3), dtype="float32")
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    a = np.asarray(a, dtype="float32")
    R, G, B = background
    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B
    return rgb
