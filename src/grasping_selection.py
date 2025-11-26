"""
Grasp selection for catching thrown objects.

Uses Darboux frame-based grasp sampling with collision checking and
cost-based ranking to select optimal grasps within the robot's reachable workspace.

Reference: https://arxiv.org/pdf/1706.09911.pdf
"""

import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import KDTree

from pydrake.all import (
    AbstractValue,
    AddMultibodyPlantSceneGraph,
    Box,
    DiagramBuilder,
    InverseKinematics,
    LeafSystem,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    PointCloud,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Solve,
)
from manipulation.meshcat_utils import AddMeshcatTriad
from manipulation.scenarios import AddMultibodyTriad
from manipulation.utils import ConfigureParser

from utils import ObjectTrajectory


@dataclass
class GraspCostResult:
    """Result of grasp cost computation."""
    total_cost: float
    centroid_distance: float  # distance from gripper y-axis to object centroid
    direction_cost: float     # how well gripper points away from robot
    alignment_cost: float     # how well gripper aligns with object velocity


@dataclass
class WorkspaceConfig:
    """Configuration for robot workspace bounds."""
    min_radius: float = 0.42  # minimum XY distance from robot base (m)
    max_radius: float = 0.75  # maximum XY distance from robot base (m)
    catch_time_ratio: float = 0.475  # where in reachable window to catch (0-1)


@dataclass
class GraspThresholds:
    """Thresholds for filtering grasp candidates."""
    centroid_distance: float = 0.030  # max distance to object centroid
    direction: float = 0.400          # max direction cost
    alignment: float = 0.100          # max alignment cost


class GraspSelector(LeafSystem):
    """
    Samples and selects optimal grasps for catching a thrown object.
    
    Pipeline:
    1. Wait for object trajectory prediction
    2. Determine when object is in robot's reachable workspace
    3. Sample grasp candidates using Darboux frames on object surface
    4. Filter by collision, reachability, and cost thresholds
    5. Select lowest-cost grasp and track it as object trajectory updates
    """
    
    # Gripper closing region bounds (in gripper frame)
    CLOSING_REGION_MIN = np.array([-0.05, 0.05, -0.00625])
    CLOSING_REGION_MAX = np.array([0.05, 0.1125, 0.00625])
    
    def __init__(self, plant, scene_graph, meshcat, thrown_model_name, grasp_random_seed):
        LeafSystem.__init__(self)
        
        print("Using grasp selector version 2")
        
        self.plant = plant
        self.scene_graph = scene_graph
        self.meshcat = meshcat
        self.thrown_model_name = thrown_model_name
        self.grasp_random_seed = grasp_random_seed
        self.visualize = True
        
        # Config
        self._workspace = WorkspaceConfig()
        self._thresholds = GraspThresholds()
        
        # Visualization offset for grasp candidates
        self._viz_offset = RigidTransform([-1, -1, 1])
        
        # Cached state (set after first grasp selection)
        self._selected_grasp_obj_frame: Optional[RigidTransform] = None
        self._obj_catch_time: Optional[float] = None
        self._obj_pose_at_catch: Optional[RigidTransform] = None
        self._obj_traj: Optional[ObjectTrajectory] = None
        self._obj_pc: Optional[PointCloud] = None
        
        # Input ports
        self.DeclareAbstractInputPort("object_pc", AbstractValue.Make(PointCloud()))
        self.DeclareAbstractInputPort("object_trajectory", AbstractValue.Make(ObjectTrajectory()))
        
        # Output port
        port = self.DeclareAbstractOutputPort(
            "grasp_selection",
            lambda: AbstractValue.Make({RigidTransform(): 0}),
            self.SelectGrasp,
        )
        port.disable_caching_by_default()

    # =========================================================================
    # Main Entry Point
    # =========================================================================
    
    def SelectGrasp(self, context, output):
        """Output callback: select grasp on first call, update on subsequent calls."""
        if self._selected_grasp_obj_frame is None:
            self._compute_initial_grasp(context, output)
        else:
            self._update_grasp_for_trajectory(context, output)

    def _compute_initial_grasp(self, context, output):
        """First-time grasp computation."""
        # Get inputs
        self._obj_pc = self.get_input_port(0).Eval(context).VoxelizedDownSample(voxel_size=0.0025)
        self._obj_pc.EstimateNormals(0.05, 30)
        self._obj_traj = self.get_input_port(1).Eval(context)
        
        # Check if trajectory is ready
        if self._obj_traj == ObjectTrajectory():
            print("received default traj (in SelectGrasp)")
            return
        
        # Visualize point cloud
        self.meshcat.SetObject("cloud", self._obj_pc)
        obj_pc_centroid = np.mean(self._obj_pc.xyzs(), axis=1)
        
        # Find catch time within reachable workspace
        self._obj_catch_time = self._compute_catch_time()
        self._obj_pose_at_catch = self._obj_traj.value(self._obj_catch_time)
        
        # Sample and rank grasp candidates
        start = time.time()
        grasp_candidates = self._compute_candidate_grasps(obj_pc_centroid)
        print(f"-----------grasp sampling runtime: {time.time() - start}")
        
        # Visualize point cloud offset
        if self.visualize:
            obj_pc_viz = PointCloud(self._obj_pc)
            obj_pc_viz.mutable_xyzs()[:] = self._viz_offset @ obj_pc_viz.xyzs()
            self.meshcat.SetObject("cloud", obj_pc_viz)
        
        # Select best grasp
        best_grasp = self._select_best_grasp(grasp_candidates)
        
        if best_grasp is None:
            print("No valid grasp found!")
            return
        
        # Convert to world frame and output
        best_grasp_world = self._obj_pose_at_catch @ best_grasp
        
        if self.visualize:
            self._draw_grasp_candidate(best_grasp_world, prefix="gripper_best", apply_offset=False)
        
        output.set_value({best_grasp_world: self._obj_catch_time})
        self._selected_grasp_obj_frame = best_grasp

    def _update_grasp_for_trajectory(self, context, output):
        """Update grasp pose as trajectory prediction refines."""
        self._obj_traj = self.get_input_port(1).Eval(context)
        
        # Update position from latest trajectory, keep original rotation (less noisy)
        updated_position = self._obj_traj.value(self._obj_catch_time).translation()
        estimated_catch_pose = RigidTransform(self._obj_pose_at_catch.rotation(), updated_position)
        
        # Transform grasp to world frame
        selected_grasp_world = estimated_catch_pose @ self._selected_grasp_obj_frame
        output.set_value({selected_grasp_world: self._obj_catch_time})

    # =========================================================================
    # Workspace Analysis
    # =========================================================================
    
    def _compute_catch_time(self) -> float:
        """Find optimal catch time when object is in robot's reachable workspace."""
        search_times = np.linspace(0.5, 1.0, 20)
        
        reachable_start = 0.5
        reachable_end = 1.0
        
        # Forward search for first reachable time
        for t in search_times:
            if self._is_in_workspace(t):
                reachable_start = t
                break
        
        # Backward search for last reachable time
        for t in reversed(search_times):
            if self._is_in_workspace(t):
                reachable_end = t
                break
        
        # Return time at configured ratio through reachable window
        return self._workspace.catch_time_ratio * (reachable_start + reachable_end)

    def _is_in_workspace(self, t: float) -> bool:
        """Check if object position at time t is in robot's reachable workspace."""
        pos = self._obj_traj.value(t).translation()
        dist_sq = pos[0]**2 + pos[1]**2
        return self._workspace.min_radius**2 < dist_sq < self._workspace.max_radius**2

    # =========================================================================
    # Grasp Sampling
    # =========================================================================
    
    def _compute_candidate_grasps(self, obj_pc_centroid: np.ndarray) -> Dict[RigidTransform, float]:
        """
        Sample grasp candidates using Darboux frames and parallel evaluation.
        
        Returns:
            Dict mapping grasp poses (in object frame) to their costs
        """
        np.random.seed(self.grasp_random_seed)
        
        kdtree = KDTree(self._obj_pc.xyzs().T)
        candidate_num = 2000
        ball_radius = 0.002
        
        candidates = {}
        candidates_lock = threading.Lock()
        
        def evaluate_candidate(point_idx):
            """Thread worker to evaluate a single grasp candidate."""
            grasp = self._sample_grasp_at_point(point_idx, kdtree, ball_radius)
            if grasp is None:
                return
            
            # Compute cost and check thresholds
            cost_result = self._compute_grasp_cost(obj_pc_centroid, grasp)
            
            if not self._passes_thresholds(cost_result):
                return
            
            # Check collision and nonempty (expensive)
            if self._check_collision(grasp):
                return
            
            if not self._check_nonempty(grasp):
                return
            
            with candidates_lock:
                candidates[grasp] = cost_result.total_cost
        
        # Launch threads
        threads = []
        for _ in range(candidate_num):
            point_idx = np.random.randint(0, self._obj_pc.size())
            t = threading.Thread(target=evaluate_candidate, args=(point_idx,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        print(f"found {len(candidates)} potential grasps")
        return candidates

    def _sample_grasp_at_point(
        self, 
        point_idx: int, 
        kdtree: KDTree, 
        ball_radius: float
    ) -> Optional[RigidTransform]:
        """Sample a grasp pose using Darboux frame at a point."""
        X_OF = self._compute_darboux_frame(point_idx, kdtree, ball_radius)
        if X_OF is None:
            return None
        
        # Offset gripper from surface based on object type
        y_offset = self._get_gripper_offset()
        return X_OF @ RigidTransform([0, y_offset, 0])

    def _get_gripper_offset(self) -> float:
        """Get gripper Y-offset based on object type."""
        name_lower = self.thrown_model_name.lower()
        if "banana" in name_lower:
            return -0.04
        return -0.05  # default for ball, bottle

    def _compute_darboux_frame(
        self, 
        point_idx: int, 
        kdtree: KDTree, 
        ball_radius: float,
        max_nn: int = 50
    ) -> Optional[RigidTransform]:
        """
        Compute Darboux frame at a point on the object surface.
        
        The Darboux frame provides a natural coordinate system for grasping:
        - Y-axis: surface normal (pointing into object)
        - X-axis: major curvature direction
        - Z-axis: minor curvature direction
        """
        points = self._obj_pc.xyzs()
        normals = self._obj_pc.normals()
        
        # Find nearest neighbors
        distances, indices = kdtree.query(points[:, point_idx], max_nn, distance_upper_bound=ball_radius)
        valid_mask = np.isfinite(distances)
        nn_indices = indices[valid_mask]
        
        if len(nn_indices) < 3:
            return None
        
        # Compute normal covariance matrix
        nn_normals = normals[:, nn_indices]
        N = nn_normals @ nn_normals.T
        
        # Eigen decomposition for principal directions
        eig_vals, eig_vecs = np.linalg.eig(N)
        sorted_idx = np.argsort(eig_vals)[::-1]
        eig_vecs = eig_vecs[:, sorted_idx]
        
        # Ensure normal points into object
        v1 = eig_vecs[:, 0]  # normal direction
        if v1 @ normals[:, point_idx] > 0:
            v1 = -v1
        
        v2 = eig_vecs[:, 1]  # major tangent
        v3 = eig_vecs[:, 2]  # minor tangent
        
        # Construct rotation matrix [v2, v1, v3] for gripper frame
        R = np.column_stack([v2, v1, v3])
        
        # Fix improper rotation (reflection)
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]
        
        return RigidTransform(RotationMatrix(R), points[:, point_idx])

    # =========================================================================
    # Grasp Evaluation
    # =========================================================================
    
    def _compute_grasp_cost(
        self, 
        obj_pc_centroid: np.ndarray, 
        X_OG: RigidTransform
    ) -> GraspCostResult:
        """
        Compute cost for a grasp candidate.
        
        Cost components:
        1. Centroid distance: gripper Y-axis should pass through object center
        2. Direction: gripper should point away from robot base
        3. Alignment: gripper Z-axis should align with object velocity
        """
        # 1. Distance from gripper Y-axis to object centroid
        to_centroid = obj_pc_centroid - X_OG.translation()
        y_axis = X_OG.rotation().matrix()[:, 1]
        projection = (np.dot(to_centroid, y_axis) / np.linalg.norm(y_axis)) * y_axis
        centroid_distance = np.linalg.norm(to_centroid - projection)
        
        # Transform to world frame for remaining costs
        X_WG = self._obj_pose_at_catch @ X_OG
        
        # 2. Direction cost: Y-axis should point away from robot
        to_gripper_xy = np.array([*X_WG.translation()[:2], 0])
        to_gripper_xy = to_gripper_xy / np.linalg.norm(to_gripper_xy)
        y_axis_world = X_WG.rotation().matrix()[:, 1]
        direction_cost = 1 - np.dot(to_gripper_xy, y_axis_world)
        
        # 3. Alignment cost: Z-axis should align with object velocity
        obj_vel = self._obj_traj.EvalDerivative(self._obj_catch_time)[:3]
        obj_dir = obj_vel / np.linalg.norm(obj_vel)
        z_axis_world = X_WG.rotation().matrix()[:, 2]
        alignment_cost = 1 - np.abs(np.dot(obj_dir, z_axis_world))
        
        total = alignment_cost + direction_cost + 10 * centroid_distance
        
        return GraspCostResult(
            total_cost=total,
            centroid_distance=centroid_distance,
            direction_cost=direction_cost,
            alignment_cost=alignment_cost,
        )

    def _passes_thresholds(self, cost: GraspCostResult) -> bool:
        """Check if grasp passes all cost thresholds."""
        return (
            cost.centroid_distance <= self._thresholds.centroid_distance and
            cost.direction_cost <= self._thresholds.direction and
            cost.alignment_cost <= self._thresholds.alignment
        )

    def _select_best_grasp(
        self, 
        candidates: Dict[RigidTransform, float]
    ) -> Optional[RigidTransform]:
        """Select lowest-cost grasp from candidates."""
        if not candidates:
            return None
        
        best_grasp = None
        best_cost = float('inf')
        
        for grasp, cost in candidates.items():
            if self.visualize:
                self._draw_grasp_candidate(grasp, prefix=f"gripper_{time.time()}")
            
            if cost < best_cost:
                best_cost = cost
                best_grasp = grasp
        
        return best_grasp

    # =========================================================================
    # Collision & Validity Checks
    # =========================================================================
    
    def _check_collision(self, X_OG: RigidTransform) -> bool:
        """
        Check if gripper collides with object point cloud.
        
        Builds temporary MBP with gripper and checks signed distance.
        """
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        parser = Parser(plant)
        ConfigureParser(parser)
        parser.AddModelsFromUrl("package://manipulation/schunk_wsg_50_welded_fingers.sdf")
        AddMultibodyTriad(plant.GetFrameByName("body"), scene_graph)
        plant.Finalize()
        
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        
        plant_context = plant.GetMyContextFromRoot(context)
        scene_graph_context = scene_graph.GetMyContextFromRoot(context)
        plant.SetFreeBodyPose(plant_context, plant.GetBodyByName("body"), X_OG)
        
        query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
        
        for pt in self._obj_pc.xyzs().T:
            distances = query_object.ComputeSignedDistanceToPoint(pt)
            for dist_info in distances:
                if dist_info.distance < 0:
                    return True  # Collision
        
        return False

    def _check_nonempty(self, X_OG: RigidTransform) -> bool:
        """Check if gripper closing region contains object points."""
        X_GO = X_OG.inverse()
        points_gripper = X_GO @ self._obj_pc.xyzs()
        
        in_region = np.all([
            points_gripper[0, :] >= self.CLOSING_REGION_MIN[0],
            points_gripper[0, :] <= self.CLOSING_REGION_MAX[0],
            points_gripper[1, :] >= self.CLOSING_REGION_MIN[1],
            points_gripper[1, :] <= self.CLOSING_REGION_MAX[1],
            points_gripper[2, :] >= self.CLOSING_REGION_MIN[2],
            points_gripper[2, :] <= self.CLOSING_REGION_MAX[2],
        ], axis=0)
        
        return in_region.any()

    # =========================================================================
    # Visualization
    # =========================================================================
    
    def _draw_grasp_candidate(
        self, 
        X_G: RigidTransform, 
        prefix: str = "gripper",
        apply_offset: bool = True
    ):
        """Visualize a gripper at the given pose."""
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        parser = Parser(plant)
        ConfigureParser(parser)
        parser.AddModelsFromUrl("package://manipulation/schunk_wsg_50_welded_fingers.sdf")
        
        pose = self._viz_offset @ X_G if apply_offset else X_G
        
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body"), pose)
        AddMultibodyTriad(plant.GetFrameByName("body"), scene_graph)
        plant.Finalize()
        
        params = MeshcatVisualizerParams()
        params.prefix = prefix
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, self.meshcat, params)
        
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        diagram.ForcedPublish(context)