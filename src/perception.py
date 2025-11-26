"""
Perception systems for object tracking and trajectory prediction.

This module provides:
- Camera setup utilities for multi-view depth sensing
- Point cloud generation from depth images
- Real-time trajectory prediction using ICP and RANSAC

Output ports:
- Trajectory object (ObjectTrajectory)
- Downsampled PointCloud object (in object frame)
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple

import itertools
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree

from pydrake.all import (
    AbstractValue,
    Context,
    DiagramBuilder,
    Diagram,
    InputPort,
    ImageDepth32F,
    ImageLabel16I,
    LeafSystem,
    MakeRenderEngineGl,
    Meshcat,
    MultibodyPlant,
    OutputPort,
    PointCloud,
    Rgba,
    RgbdSensor,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    CameraConfig,
    Sphere,
    Value,
)

from utils import ObjectTrajectory


# Global counter for unique camera group naming
_camera_group_idx = 0


# =============================================================================
# Camera Setup
# =============================================================================

def add_cameras(
    builder: DiagramBuilder,
    station: Diagram,
    plant: MultibodyPlant,
    camera_width: int,
    camera_height: int,
    horizontal_num: int,
    vertical_num: int,
    camera_distance: float,
    cameras_center: npt.NDArray[np.float32],
) -> Tuple[List[RgbdSensor], List[RigidTransform]]:
    """
    Create a spherical array of RGBD cameras around a center point.
    
    Args:
        builder: DiagramBuilder to add cameras to
        station: Hardware station containing scene_graph
        plant: MultibodyPlant for world frame reference
        camera_width: Image width in pixels
        camera_height: Image height in pixels
        horizontal_num: Number of cameras around horizontal circle
        vertical_num: Number of cameras in vertical arc
        camera_distance: Distance from cameras_center to each camera
        cameras_center: 3D point cameras look toward
        
    Returns:
        Tuple of (camera_systems, camera_transforms)
    """
    global _camera_group_idx
    
    # Configure camera parameters
    camera_config = CameraConfig()
    camera_config.z_far = 30
    camera_config.width = camera_width
    camera_config.height = camera_height
    
    # Ensure renderer exists
    scene_graph = station.GetSubsystemByName("scene_graph")
    if not scene_graph.HasRenderer(camera_config.renderer_name):
        scene_graph.AddRenderer(camera_config.renderer_name, MakeRenderEngineGl())
    
    camera_systems = []
    camera_transforms = []
    
    # Generate spherical coordinates
    thetas = np.linspace(0, 2 * np.pi, horizontal_num, endpoint=False)
    phis = np.linspace(0, -np.pi, vertical_num + 2)[1:-1]  # Exclude poles
    
    for idx, (theta, phi) in enumerate(itertools.product(thetas, phis)):
        name = f"camera{idx}_group{_camera_group_idx}"
        
        # Compute camera transform: rotate around center, then offset by distance
        rotation = (
            RollPitchYaw(0, 0, theta).ToRotationMatrix() 
            @ RollPitchYaw(phi, 0, 0).ToRotationMatrix()
        )
        transform = (
            RigidTransform(rotation, cameras_center) 
            @ RigidTransform([0, 0, -camera_distance])
        )
        
        # Create and connect camera
        _, depth_camera = camera_config.MakeCameras()
        camera_sys = builder.AddSystem(RgbdSensor(
            parent_id=plant.GetBodyFrameIdIfExists(plant.world_frame().body().index()),
            X_PB=transform,
            depth_camera=depth_camera,
        ))
        camera_sys.set_name(name)
        
        builder.Connect(
            station.GetOutputPort("query_object"),
            camera_sys.query_object_input_port(),
        )
        
        # Export outputs
        builder.ExportOutput(camera_sys.color_image_output_port(), f"{name}.rgb_image")
        builder.ExportOutput(camera_sys.depth_image_32F_output_port(), f"{name}.depth_image")
        builder.ExportOutput(camera_sys.label_image_output_port(), f"{name}.label_image")
        
        camera_systems.append(camera_sys)
        camera_transforms.append(transform)
    
    _camera_group_idx += 1
    return camera_systems, camera_transforms


# =============================================================================
# Base Camera System
# =============================================================================

class CameraBackedSystem(LeafSystem):
    """
    Base class for systems that process multi-camera depth data.
    
    Provides common functionality for:
    - Managing camera input ports (depth + label images)
    - Converting depth images to 3D point clouds
    - Filtering points by object label
    """
    
    def __init__(
        self,
        cameras: List[RgbdSensor],
        camera_transforms: List[RigidTransform],
        pred_thresh: int,
        thrown_model_name: str,
        plant: MultibodyPlant,
        meshcat: Optional[Meshcat] = None,
    ):
        """
        Args:
            cameras: List of RgbdSensor cameras
            camera_transforms: World-frame transform for each camera
            pred_thresh: Minimum points per camera to include in cloud
            thrown_model_name: Name of the object model to track
            plant: MultibodyPlant containing the object
            meshcat: Optional meshcat for visualization
        """
        super().__init__()
        
        self._pred_thresh = pred_thresh
        self._camera_transforms = camera_transforms
        self._meshcat = meshcat
        
        # Get camera intrinsics
        self._camera_infos = [
            camera.default_depth_render_camera().core().intrinsics() 
            for camera in cameras
        ]
        
        # Get object body index for label filtering
        model_idx = plant.GetModelInstanceByName(thrown_model_name)
        body_idx, = map(int, plant.GetBodyIndices(model_idx))
        self._obj_idx = body_idx
        
        # Create input ports for each camera
        self._camera_depth_inputs = []
        self._camera_label_inputs = []
        
        for i, camera_info in enumerate(self._camera_infos):
            depth_port = self.DeclareAbstractInputPort(
                f"camera{i}.depth_input",
                AbstractValue.Make(ImageDepth32F(camera_info.width(), camera_info.height())),
            )
            label_port = self.DeclareAbstractInputPort(
                f"camera{i}.label_input",
                AbstractValue.Make(ImageLabel16I(camera_info.width(), camera_info.height())),
            )
            self._camera_depth_inputs.append(depth_port)
            self._camera_label_inputs.append(label_port)

    def camera_input_ports(self, camera_idx: int) -> Tuple[InputPort, InputPort]:
        """Get (depth, label) input ports for a specific camera."""
        return self._camera_depth_inputs[camera_idx], self._camera_label_inputs[camera_idx]

    def ConnectCameras(self, builder: DiagramBuilder, cameras: List[RgbdSensor]):
        """Connect camera outputs to this system's inputs."""
        for camera, depth_input, label_input in zip(
            cameras, self._camera_depth_inputs, self._camera_label_inputs
        ):
            builder.Connect(camera.depth_image_32F_output_port(), depth_input)
            builder.Connect(camera.label_image_output_port(), label_input)

    def GetCameraPoints(self, context: Context) -> npt.NDArray[np.float32]:
        """
        Get aggregated 3D point cloud from all cameras.
        
        Filters points to only include the tracked object based on label images.
        Transforms points to world frame.
        
        Returns:
            (3, N) array of 3D points in world frame
        """
        all_points = []
        
        for camera_info, transform, depth_input, label_input in zip(
            self._camera_infos,
            self._camera_transforms,
            self._camera_depth_inputs,
            self._camera_label_inputs,
        ):
            points = self._process_single_camera(
                context, camera_info, transform, depth_input, label_input
            )
            if points is not None:
                all_points.append(points)
        
        if not all_points:
            return np.zeros((3, 0))
        
        return np.concatenate(all_points, axis=1)

    def _process_single_camera(
        self,
        context: Context,
        camera_info,
        transform: RigidTransform,
        depth_input: InputPort,
        label_input: InputPort,
    ) -> Optional[npt.NDArray[np.float32]]:
        """
        Process depth image from a single camera to 3D points.
        
        Returns:
            (3, N) array of points in world frame, or None if insufficient points
        """
        height = camera_info.height()
        width = camera_info.width()
        
        depth_img = depth_input.Eval(context).data
        label_img = label_input.Eval(context).data
        
        # NOTE: If camera produces upside-down images on your machine, use:
        # depth_img = depth_input.Eval(context).data[::-1]
        # label_img = label_input.Eval(context).data[::-1]
        
        # Create pixel coordinate grids
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height), copy=False)
        
        # Filter to object pixels with valid depth
        mask = (label_img[:, :, 0] == self._obj_idx) & (np.abs(depth_img[:, :, 0]) != np.inf)
        
        if mask.sum() < self._pred_thresh:
            return None
        
        # Extract filtered coordinates
        u = u_coords[mask]
        v = v_coords[mask]
        z = depth_img[:, :, 0][mask]
        
        # Back-project to 3D (camera frame)
        x = (u - camera_info.center_x()) * z / camera_info.focal_x()
        y = (v - camera_info.center_y()) * z / camera_info.focal_y()
        
        points_camera = np.stack([x, y, z])  # (3, N)
        
        # Transform to world frame
        return transform @ points_camera

    def PublishMeshcat(self, points: npt.NDArray[np.float32]):
        """Visualize point cloud in meshcat."""
        if self._meshcat is None:
            return
            
        cloud = PointCloud(points.shape[1])
        if points.shape[1] > 0:
            cloud.mutable_xyzs()[:] = points
        
        self._meshcat.SetObject(
            f"{self.__class__.__name__}PointCloud",
            cloud,
            point_size=0.01,
            rgba=Rgba(1, 0.5, 0.5),
        )


# =============================================================================
# Point Cloud Generator
# =============================================================================

class PointCloudGenerator(CameraBackedSystem):
    """
    Captures a static point cloud of the object for ICP reference.
    
    Call CapturePointCloud() once when object is at known position to
    create the reference point cloud used by TrajectoryPredictor.
    """
    
    def __init__(
        self,
        cameras: List[RgbdSensor],
        camera_transforms: List[RigidTransform],
        cameras_center: npt.NDArray[np.float32],
        pred_thresh: int,
        thrown_model_name: str,
        plant: MultibodyPlant,
        meshcat: Optional[Meshcat] = None,
    ):
        super().__init__(
            cameras=cameras,
            camera_transforms=camera_transforms,
            pred_thresh=pred_thresh,
            thrown_model_name=thrown_model_name,
            plant=plant,
            meshcat=meshcat,
        )
        
        self._cameras_center = np.array(cameras_center)
        self._point_cloud = PointCloud()
        
        self._point_cloud_output = self.DeclareAbstractOutputPort(
            "point_cloud",
            lambda: AbstractValue.Make(PointCloud()),
            self._output_point_cloud,
        )

    @property
    def point_cloud_output_port(self) -> OutputPort:
        return self._point_cloud_output

    def _output_point_cloud(self, context: Context, output: Value):
        output.set_value(self._point_cloud)

    def CapturePointCloud(self, context: Context):
        """
        Capture current camera views as the reference point cloud.
        
        Points are transformed to be centered at origin (object frame).
        """
        points = self.GetCameraPoints(context)
        
        # Center the point cloud
        points_centered = points - self._cameras_center.reshape(3, 1)
        
        self._point_cloud = PointCloud(points_centered.shape[1])
        self._point_cloud.mutable_xyzs()[:] = points_centered
        
        if self._meshcat is not None:
            self.PublishMeshcat(points_centered)


# =============================================================================
# Trajectory Predictor
# =============================================================================

@dataclass
class ICPConfig:
    """Configuration for ICP algorithm."""
    max_iterations: int = 100
    convergence_threshold: float = 1e-10


@dataclass
class RANSACConfig:
    """Configuration for RANSAC trajectory fitting."""
    iterations: int = 20
    position_threshold: float = 0.01
    rotation_threshold: float = 0.1
    window_size: int = 30


# Counter for unique visualization names
_pred_traj_calls = 0


class TrajectoryPredictor(CameraBackedSystem):
    """
    Predicts object trajectory using ICP tracking and RANSAC fitting.
    
    Pipeline:
    1. Get current object point cloud from cameras
    2. Run ICP to estimate current pose relative to reference cloud
    3. Accumulate pose history with timestamps
    4. Fit ballistic trajectory using RANSAC
    5. Output predicted trajectory
    """
    
    def __init__(
        self,
        cameras: List[RgbdSensor],
        camera_transforms: List[RigidTransform],
        pred_thresh: int,
        pred_samples_thresh: int,
        ransac_iters: int,
        ransac_thresh: float,
        ransac_rot_thresh: float,
        ransac_window: int,
        thrown_model_name: str,
        plant: MultibodyPlant,
        estimate_pose: bool = True,
        meshcat: Optional[Meshcat] = None,
    ):
        """
        Args:
            pred_samples_thresh: Minimum pose samples before outputting trajectory
            ransac_iters: Number of RANSAC iterations
            ransac_thresh: Position inlier threshold (meters)
            ransac_rot_thresh: Rotation inlier threshold (radians)
            ransac_window: Number of recent poses to keep
            estimate_pose: If False, only estimate position (ignore rotation)
        """
        super().__init__(
            cameras=cameras,
            camera_transforms=camera_transforms,
            pred_thresh=pred_thresh,
            thrown_model_name=thrown_model_name,
            plant=plant,
            meshcat=meshcat,
        )
        
        print("Using perception version 2")
        
        self._pred_samples_thresh = pred_samples_thresh
        self._estimate_pose = estimate_pose
        
        self._icp_config = ICPConfig()
        self._ransac_config = RANSACConfig(
            iterations=ransac_iters,
            position_threshold=ransac_thresh,
            rotation_threshold=ransac_rot_thresh,
            window_size=ransac_window,
        )
        
        # Reference point cloud KD-tree (initialized lazily)
        self._point_kd_tree: Optional[KDTree] = None
        
        # Input port for reference point cloud
        self._obj_point_cloud_input = self.DeclareAbstractInputPort(
            "obj_point_cloud",
            AbstractValue.Make(PointCloud()),
        )
        
        # State: pose history and current trajectory
        self._poses_state = self.DeclareAbstractState(AbstractValue.Make(deque()))
        self._traj_state = self.DeclareAbstractState(AbstractValue.Make(ObjectTrajectory()))
        
        # Periodic update
        self.DeclarePeriodicPublishEvent(0.01, 0, self._predict_trajectory)
        
        # Output port
        self.DeclareAbstractOutputPort(
            "object_trajectory",
            lambda: AbstractValue.Make(ObjectTrajectory()),
            self._output_trajectory,
        )

    @property
    def point_cloud_input_port(self) -> OutputPort:
        return self._obj_point_cloud_input

    def _output_trajectory(self, context: Context, output: Value):
        output.set_value(context.get_abstract_state(self._traj_state).get_value())

    def _predict_trajectory(self, context: Context):
        """Main update: track object and fit trajectory."""
        scene_points = self.GetCameraPoints(context)
        
        if self._meshcat is not None:
            self.PublishMeshcat(scene_points)
        
        if scene_points.shape[1] == 0:
            return
        
        # Run ICP to get current pose
        previous_pose = self._get_previous_pose(context)
        current_pose = self._run_icp(context, scene_points.T, previous_pose.inverse())
        
        # Update RANSAC with new pose
        self._update_ransac(context, current_pose)
        
        # Visualize current detection
        self._visualize_detection(current_pose)

    def _get_previous_pose(self, context: Context) -> RigidTransform:
        """Get most recent pose from history."""
        poses = context.get_abstract_state(self._poses_state).get_value()
        if not poses:
            return RigidTransform()
        pose, _ = poses[0]
        return pose

    def _maybe_init_point_cloud(self, context: Context):
        """Lazily initialize KD-tree from reference point cloud."""
        if self._point_kd_tree is None:
            points = self._obj_point_cloud_input.Eval(context).xyzs()
            self._point_kd_tree = KDTree(points.T)

    def _run_icp(
        self,
        context: Context,
        scene_points: npt.NDArray[np.float32],
        X_init: RigidTransform = RigidTransform(),
    ) -> RigidTransform:
        """
        Run Iterative Closest Point to align scene points to reference.
        
        Args:
            scene_points: (N, 3) array of observed points
            X_init: Initial transform guess
            
        Returns:
            Estimated object pose in world frame
        """
        self._maybe_init_point_cloud(context)
        
        # Transform points by initial guess
        p_scene = (X_init @ scene_points.T).T
        X_accumulated = X_init
        
        prev_cost = np.inf
        
        for _ in range(self._icp_config.max_iterations):
            # Find closest points in reference cloud
            distances, indices = self._point_kd_tree.query(p_scene)
            cost = distances.mean()
            
            # Check convergence
            if np.allclose(prev_cost, cost, atol=self._icp_config.convergence_threshold):
                break
            prev_cost = cost
            
            # Get corresponding reference points
            p_reference = self._point_kd_tree.data[indices]
            
            # Compute centroids
            centroid_ref = p_reference.mean(axis=0)
            centroid_scene = p_scene.mean(axis=0)
            
            # Compute optimal rotation via SVD
            W = (p_reference - centroid_ref).T @ (p_scene - centroid_scene)
            U, _, Vh = np.linalg.svd(W)
            
            # Handle reflection case
            D = np.diag([1, 1, np.linalg.det(U @ Vh)])
            R_optimal = U @ D @ Vh
            
            # Compute optimal translation
            t_optimal = centroid_ref - R_optimal @ centroid_scene
            
            X_step = RigidTransform(RotationMatrix(R_optimal), t_optimal)
            
            # Update accumulated transform and points
            p_scene = (X_step @ p_scene.T).T
            X_accumulated = X_step @ X_accumulated
        
        # Invert to get object pose (not scene-to-reference transform)
        X_result = X_accumulated.inverse()
        
        # Optionally ignore rotation
        if not self._estimate_pose:
            X_result.set_rotation(RotationMatrix())
        
        return X_result

    def _update_ransac(self, context: Context, current_pose: RigidTransform):
        """
        Update pose history and fit trajectory using RANSAC.
        
        Maintains a sliding window of recent poses and fits a ballistic
        trajectory to the inliers.
        """
        # Add current pose to history
        poses_state = context.get_abstract_state(self._poses_state)
        poses = poses_state.get_mutable_value()
        poses.appendleft((current_pose, context.get_time()))
        
        # Trim to window size
        while len(poses) > self._ransac_config.window_size:
            poses.pop()
        
        # Need minimum samples for trajectory fitting
        if len(poses) < max(self._pred_samples_thresh, 2):
            return
        
        best_trajectory = ObjectTrajectory()
        best_inlier_count = 0
        best_inlier_cost = np.inf
        
        poses_list = list(poses)
        
        for _ in range(self._ransac_config.iterations):
            # Sample two random poses
            i, j = np.random.choice(len(poses_list), size=2, replace=False)
            pose_i, time_i = poses_list[i]
            pose_j, time_j = poses_list[j]
            
            # Fit trajectory through these two points
            candidate_traj = ObjectTrajectory.CalculateTrajectory(pose_i, time_i, pose_j, time_j)
            
            # Count inliers
            inlier_mask, costs = self._evaluate_trajectory(poses_list, candidate_traj)
            inlier_count = inlier_mask.sum()
            inlier_cost = costs[inlier_mask].sum()
            
            # Update best if improved
            if (inlier_count > best_inlier_count or 
                (inlier_count == best_inlier_count and inlier_cost < best_inlier_cost)):
                best_trajectory = candidate_traj
                best_inlier_count = inlier_count
                best_inlier_cost = inlier_cost
        
        # Visualize trajectory
        self._visualize_trajectory(best_trajectory)
        
        # Store result
        context.SetAbstractState(self._traj_state, best_trajectory)

    def _evaluate_trajectory(
        self,
        poses: List[Tuple[RigidTransform, float]],
        trajectory: ObjectTrajectory,
    ) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float32]]:
        """
        Evaluate how well a trajectory fits the pose history.
        
        Returns:
            Tuple of (inlier_mask, costs) arrays
        """
        position_errors = []
        rotation_errors = []
        
        for pose, time in poses:
            predicted = trajectory.value(time)
            
            pos_err = np.linalg.norm(pose.translation() - predicted.translation())
            
            rot_diff = pose.rotation().inverse() @ predicted.rotation()
            rot_err = np.abs(RotationMatrix(rot_diff).ToAngleAxis().angle())
            
            position_errors.append(pos_err)
            rotation_errors.append(rot_err)
        
        position_errors = np.array(position_errors)
        rotation_errors = np.array(rotation_errors)
        
        inlier_mask = (
            (position_errors < self._ransac_config.position_threshold) &
            (rotation_errors < self._ransac_config.rotation_threshold)
        )
        
        costs = position_errors + rotation_errors
        
        return inlier_mask, costs

    def _visualize_detection(self, pose: RigidTransform):
        """Visualize current pose detection."""
        if self._meshcat is None:
            return
            
        global _pred_traj_calls
        _pred_traj_calls += 1
        
        self._meshcat.SetObject(
            f"PredTrajSpheres/{_pred_traj_calls}",
            Sphere(0.005),
            Rgba(159 / 255, 131 / 255, 3 / 255, 1),
        )
        self._meshcat.SetTransform(f"PredTrajSpheres/{_pred_traj_calls}", pose)

    def _visualize_trajectory(self, trajectory: ObjectTrajectory):
        """Visualize predicted trajectory as series of spheres."""
        if self._meshcat is None:
            return
            
        for t in np.linspace(0, 1, 100):
            self._meshcat.SetObject(
                f"RansacSpheres/{t}",
                Sphere(0.01),
                Rgba(0.2, 0.2, 1, 1),
            )
            self._meshcat.SetTransform(
                f"RansacSpheres/{t}",
                RigidTransform(trajectory.value(t)),
            )