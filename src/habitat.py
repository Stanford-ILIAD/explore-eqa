import numpy as np
import quaternion
import habitat_sim
from .geom import get_collision_distance
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis, quat_from_two_vectors, quat_to_angle_axis


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

def make_semantic_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
    sim_cfg.load_semantic_mesh = True

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

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.hfov = settings["hfov"]

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]

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


def get_quaternion(angle, camera_tilt):
    normalized_angle = angle % (2 * np.pi)
    if np.abs(normalized_angle - np.pi) < 1e-6:
        return quat_to_coeffs(
            quaternion.quaternion(0, 0, 1, 0) *
            quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()

    return quat_to_coeffs(
        quat_from_angle_axis(angle, np.array([0, 1, 0])) *
        quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
    ).tolist()


def get_navigable_point_from(pos_start, pathfinder, max_search=1000, min_dist=6, prev_start_positions=None, min_dist_from_prev=3):
    pos_end = None
    path_points = None
    max_distance_history = -1

    pos_end_with_prev = None
    path_points_with_prev = None
    max_distance_history_with_prev = -1

    # if no previous start positions, just find a random navigable point that is the farthest among the searches
    # otherwise, find the point that is the farthest from the previous start positions, and also satisfies the min_dist
    try_count = 0
    while True:
        try_count += 1
        if try_count > max_search:
            break

        pos_end_current = pathfinder.get_random_navigable_point()
        if np.abs(pos_end_current[1] - pos_start[1]) > 0.4:  # make sure the end point is on the same level
            continue

        path = habitat_sim.ShortestPath()
        path.requested_start = pos_start
        path.requested_end = pos_end_current
        found_path = pathfinder.find_path(path)
        if found_path:
            if path.geodesic_distance > max_distance_history:
                max_distance_history = path.geodesic_distance
                pos_end = pos_end_current
                path_points = path.points

            if prev_start_positions is not None:
                if np.all(
                    np.linalg.norm(
                        prev_start_positions - pos_end_current, axis=1
                    ) > min_dist_from_prev
                ):
                    if path.geodesic_distance > max_distance_history_with_prev:
                        max_distance_history_with_prev = path.geodesic_distance
                        pos_end_with_prev = pos_end_current
                        path_points_with_prev = path.points

        if found_path and max_distance_history > min_dist and prev_start_positions is None:
            break
        if found_path and max_distance_history_with_prev > min_dist and prev_start_positions is not None:
            break

    # satiny check
    if pos_end is not None and path_points is not None:
        assert np.array_equal(path_points[0], np.asarray(pos_start, dtype=np.float32)) and np.array_equal(path_points[-1], pos_end)
    if pos_end_with_prev is not None and path_points_with_prev is not None:
        assert np.array_equal(path_points_with_prev[0], np.asarray(pos_start, dtype=np.float32)) and np.array_equal(path_points_with_prev[-1], pos_end_with_prev)

    if prev_start_positions is None:
        return pos_end, path_points, max_distance_history
    else:
        if pos_end_with_prev is not None and path_points_with_prev is not None and max_distance_history_with_prev > min_dist:
            print(f'Found point success!!!!!!!!!!!!!!!!!!!!!!!')
            return pos_end_with_prev, path_points_with_prev, max_distance_history_with_prev
        else:
            # if no point is found that satisfies the min_dist, just return the farthest point
            print(f'No point found that satisfies the min_dist, return the farthest point!!!!!!!!!!!!!!!!!!!!!!!!!!')
            return pos_end, path_points, max_distance_history


def get_navigable_point_from_new(pos_start, pathfinder, max_search=1000, min_dist=6, prev_start_positions=None, min_dist_from_prev=3, max_samples=100):
    if min_dist < 0:
        return None, None, None

    pos_end_list = []

    # if no previous start positions, just find a random navigable point that is the farthest among the searches
    # otherwise, find the point that is the farthest from the previous start positions, and also satisfies the min_dist
    try_count = 0
    while True:
        try_count += 1
        if try_count > max_search:
            break

        pos_end_current = pathfinder.get_random_navigable_point()
        if np.abs(pos_end_current[1] - pos_start[1]) > 0.4:  # make sure the end point is on the same level
            continue

        path = habitat_sim.ShortestPath()
        path.requested_start = pos_start
        path.requested_end = pos_end_current
        found_path = pathfinder.find_path(path)
        if found_path:
            if path.geodesic_distance > min_dist:
                if prev_start_positions is None:
                    pos_end_list.append(pos_end_current)
                else:
                    if np.all(
                        np.linalg.norm(
                            prev_start_positions - pos_end_current, axis=1
                        ) > min_dist_from_prev
                    ):
                        pos_end_list.append(pos_end_current)

        if len(pos_end_list) >= max_samples:
            break

    if len(pos_end_list) == 0:
        # if no point is found that satisfies the min_dist, then find again with shorter min_dist
        return get_navigable_point_from_new(
            pos_start,
            pathfinder,
            max_search,
            min_dist - 2,
            prev_start_positions,
            min_dist_from_prev,
            max_samples
        )

    # sample a ramdom point from the list
    while True:
        pos_end = pos_end_list[np.random.randint(len(pos_end_list))]
        path = habitat_sim.ShortestPath()
        path.requested_start = pos_start
        path.requested_end = pos_end
        found_path = pathfinder.find_path(path)
        if found_path:
            break
    return pos_end, path.points, path.geodesic_distance


def get_navigable_point_to(pos_end, pathfinder, max_search=1000, min_dist=6, prev_start_positions=None):
    pos_start, path_point, travel_dist = get_navigable_point_from(
        pos_end, pathfinder, max_search, min_dist, prev_start_positions
    )
    if pos_start is None or path_point is None:
        return None, None, None

    # reverse the path_point
    path_point = path_point[::-1]
    return pos_start, path_point, travel_dist


def get_navigable_point_to_new(pos_end, pathfinder, max_search=1000, min_dist=6, prev_start_positions=None):
    pos_start, path_point, travel_dist = get_navigable_point_from_new(
        pos_end, pathfinder, max_search, min_dist, prev_start_positions
    )
    if pos_start is None or path_point is None:
        return None, None, None

    # reverse the path_point
    path_point = path_point[::-1]
    return pos_start, path_point, travel_dist


def get_frontier_observation(
        agent, simulator, cfg, tsdf_planner,
        view_frontier_direction, init_pts, camera_tilt, max_try_count=10
):
    agent_state = habitat_sim.AgentState()

    # solve edge cases of viewing direction
    default_view_direction = np.asarray([0., 0., -1.])
    if np.linalg.norm(view_frontier_direction) < 1e-3:
        view_frontier_direction = default_view_direction
    view_frontier_direction = view_frontier_direction / np.linalg.norm(view_frontier_direction)

    # convert view direction to voxel space
    # since normal to voxel is just scale and shift, we don't need that conversion
    view_dir_voxel = pos_habitat_to_normal(view_frontier_direction)[:2]

    # set agent observation direction
    if np.dot(view_frontier_direction, default_view_direction) / np.linalg.norm(view_frontier_direction) < -1 + 1e-3:
        # if the rotation is to rotate 180 degree, then the quaternion is not unique
        # we need to specify rotating along y-axis
        agent_state.rotation = quat_to_coeffs(
            quaternion.quaternion(0, 0, 1, 0)
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()
    else:
        agent_state.rotation = quat_to_coeffs(
            quat_from_two_vectors(default_view_direction, view_frontier_direction)
            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
        ).tolist()

    occupied_map = tsdf_planner.occupied_map_camera

    try_count = 0
    pts = init_pts.copy()
    valid_observation = None
    while try_count < max_try_count:
        try_count += 1

        # check whether current view is valid
        collision_dist = tsdf_planner._voxel_size * get_collision_distance(
            occupied_map,
            pos=tsdf_planner.habitat2voxel(pts),
            direction=view_dir_voxel
        )

        if collision_dist >= cfg.collision_dist:
            # set agent state
            agent_state.position = pts

            # get observations
            agent.set_state(agent_state)
            obs = simulator.get_sensor_observations()
            rgb = obs["color_sensor"]
            depth = obs["depth_sensor"]
            semantic_obs = obs["semantic_sensor"]

            # check whether the observation is valid
            keep_observation = True
            black_pix_ratio = np.sum(semantic_obs == 0) / (cfg.img_height * cfg.img_width)
            if black_pix_ratio > cfg.black_pixel_ratio_frontier:
                keep_observation = False
            positive_depth = depth[depth > 0]
            if positive_depth.size == 0 or np.percentile(positive_depth, 30) < cfg.min_30_percentile_depth:
                keep_observation = False
            if keep_observation:
                valid_observation = rgb
                break

        # update pts
        pts = pts - view_frontier_direction * tsdf_planner._voxel_size

    if valid_observation is not None:
        return valid_observation

    # if no valid observation is found, then just get the observation at init position
    agent_state.position = init_pts
    agent.set_state(agent_state)
    obs = simulator.get_sensor_observations()
    return obs["color_sensor"]


















