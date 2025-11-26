"""
Object Catching System - Main Entry Point

Simulates a robot arm catching thrown objects using:
- Multi-camera perception for trajectory prediction
- Grasp selection using Darboux frames
- Real-time trajectory optimization for motion planning

Usage:
    python main.py --obj [t|b|p] --distance [c|f] --randomization [seed]
    
    --obj: t=tennis ball, b=banana, p=pill bottle
    --distance: c=close, f=far
    --randomization: integer seed for grasp sampling
"""

import argparse
import os

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import (
    DiagramBuilder,
    InverseDynamicsController,
    ModelInstanceIndex,
    MultibodyPlant,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
)
from manipulation.scenarios import AddIiwa

from station import MakeHardwareStation, load_scenario
from utils import (
    diagram_visualize_connections,
    throw_object_close,
    throw_object_far,
    calculate_obj_distance_to_gripper,
)
from perception import PointCloudGenerator, TrajectoryPredictor, add_cameras
from grasping_selection import GraspSelector
from motion_planner import MotionPlanner


# =============================================================================
# Configuration
# =============================================================================

SCENARIO_FILES = {
    't': ("Tennis Ball", "data/scenario_tennis_ball.yaml"),
    'b': ("Banana", "data/scenario_banana.yaml"),
    'p': ("Pill Bottle", "data/scenario_pill_bottle.yaml"),
}

# Object rotations for throwing (close distance)
THROW_ROTATIONS_CLOSE = {
    'ball': RotationMatrix(),
    'banana': (
        RotationMatrix.MakeZRotation(-np.pi / 4) 
        @ RotationMatrix.MakeXRotation(-np.pi / 4) 
        @ RotationMatrix.MakeZRotation(-np.pi / 2)
    ),
    'bottle': (
        RotationMatrix.MakeZRotation(-np.pi / 6) 
        @ RotationMatrix.MakeXRotation(np.pi / 3.8)
    ),
}

# Object rotations for throwing (far distance)
THROW_ROTATIONS_FAR = {
    'ball': RotationMatrix(),
    'banana': (
        RotationMatrix.MakeZRotation(-np.pi / 4) 
        @ RotationMatrix.MakeXRotation(-np.pi / 3.6) 
        @ RotationMatrix.MakeZRotation(-np.pi / 2)
    ),
    'bottle': (
        RotationMatrix.MakeZRotation(-np.pi / 6) 
        @ RotationMatrix.MakeXRotation(np.pi / 6)
    ),
}

OBJECT_BODY_NAMES = ["Tennis_ball", "Banana", "pill_bottle"]


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Object catching simulation")
    parser.add_argument(
        '--obj', 
        choices=['t', 'b', 'p'],
        default='t',
        help="Object to throw: t=tennis ball, b=banana, p=pill bottle"
    )
    parser.add_argument(
        '--distance',
        choices=['c', 'f'],
        default='c', 
        help="Throw distance: c=close, f=far"
    )
    parser.add_argument(
        '--randomization',
        type=int,
        default=0,
        help="Random seed for grasp sampling"
    )
    return parser.parse_args()


def get_scenario_config(args):
    """Get scenario configuration from arguments."""
    obj_name, scenario_file = SCENARIO_FILES.get(args.obj, SCENARIO_FILES['t'])
    print(f"Throwing {obj_name}")
    
    if args.distance == 'f':
        print("Throwing object from far distance.")
        throw_distance = "far"
    else:
        print("Throwing object from close distance.")
        throw_distance = "close"
    
    print(f"Randomization: {args.randomization}")
    
    return scenario_file, throw_distance, args.randomization


# =============================================================================
# System Building
# =============================================================================

def find_thrown_object(plant) -> str:
    """Find the model instance name that starts with 'obj'."""
    for model_idx in range(plant.num_model_instances()):
        model_name = plant.GetModelInstanceName(ModelInstanceIndex(model_idx))
        if model_name.startswith('obj'):
            return model_name
    raise ValueError("No object found with 'obj' prefix")


def setup_cameras(builder, station, plant, point_cloud_center):
    """Setup ICP tracking cameras and point cloud capture cameras."""
    # ICP cameras: lower resolution, wider coverage for tracking
    icp_cameras, icp_transforms = add_cameras(
        builder=builder,
        station=station,
        plant=plant,
        camera_width=400,
        camera_height=300,
        horizontal_num=4,
        vertical_num=5,
        camera_distance=7,
        cameras_center=[0, 0, 0],
    )
    
    # Point cloud cameras: higher resolution, focused on capture position
    pc_cameras, pc_transforms = add_cameras(
        builder=builder,
        station=station,
        plant=plant,
        camera_width=800,
        camera_height=600,
        horizontal_num=8,
        vertical_num=4,
        camera_distance=1,
        cameras_center=point_cloud_center,
    )
    
    return icp_cameras, icp_transforms, pc_cameras, pc_transforms


def setup_perception(builder, icp_cameras, icp_transforms, pc_cameras, pc_transforms,
                     point_cloud_center, obj_name, plant, meshcat):
    """Setup point cloud generation and trajectory prediction systems."""
    # Point cloud generator
    point_cloud_system = builder.AddSystem(PointCloudGenerator(
        cameras=pc_cameras,
        camera_transforms=pc_transforms,
        cameras_center=point_cloud_center,
        pred_thresh=5,
        thrown_model_name=obj_name,
        plant=plant,
    ))
    point_cloud_system.ConnectCameras(builder, pc_cameras)
    
    # Trajectory predictor
    traj_predictor = builder.AddSystem(TrajectoryPredictor(
        cameras=icp_cameras,
        camera_transforms=icp_transforms,
        pred_thresh=5,
        pred_samples_thresh=6,
        thrown_model_name=obj_name,
        ransac_iters=20,
        ransac_thresh=0.01,
        ransac_rot_thresh=0.1,
        ransac_window=30,
        plant=plant,
        estimate_pose=("ball" not in obj_name),
        meshcat=meshcat,
    ))
    traj_predictor.ConnectCameras(builder, icp_cameras)
    
    # Connect point cloud to trajectory predictor
    builder.Connect(
        point_cloud_system.GetOutputPort("point_cloud"),
        traj_predictor.point_cloud_input_port,
    )
    
    return point_cloud_system, traj_predictor


def setup_planning(builder, station, plant, scene_graph, meshcat, 
                   traj_predictor, point_cloud_system, obj_name, grasp_seed):
    """Setup grasp selection and motion planning systems."""
    # Grasp selector
    grasp_selector = builder.AddSystem(
        GraspSelector(plant, scene_graph, meshcat, obj_name, grasp_seed)
    )
    builder.Connect(
        traj_predictor.GetOutputPort("object_trajectory"),
        grasp_selector.GetInputPort("object_trajectory"),
    )
    builder.Connect(
        point_cloud_system.GetOutputPort("point_cloud"),
        grasp_selector.GetInputPort("object_pc"),
    )
    
    # Motion planner
    motion_planner = builder.AddSystem(MotionPlanner(plant, meshcat))
    builder.Connect(
        grasp_selector.GetOutputPort("grasp_selection"),
        motion_planner.GetInputPort("grasp_selection"),
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        motion_planner.GetInputPort("iiwa_current_pose"),
    )
    builder.Connect(
        traj_predictor.GetOutputPort("object_trajectory"),
        motion_planner.GetInputPort("object_trajectory"),
    )
    builder.Connect(
        station.GetOutputPort("iiwa_state"),
        motion_planner.GetInputPort("iiwa_state"),
    )
    builder.Connect(
        motion_planner.GetOutputPort("wsg_command"),
        station.GetInputPort("wsg.position"),
    )
    
    return grasp_selector, motion_planner


def setup_controller(builder, station, motion_planner):
    """Setup inverse dynamics controller for the iiwa arm."""
    # Create controller plant
    controller_plant = MultibodyPlant(time_step=0.001)
    AddIiwa(controller_plant)
    controller_plant.Finalize()
    
    num_positions = controller_plant.num_positions()
    
    # PD gains: kp=300, ki=1, kd=20
    controller = builder.AddSystem(InverseDynamicsController(
        controller_plant,
        kp=[300] * num_positions,
        ki=[1] * num_positions,
        kd=[20] * num_positions,
        has_reference_acceleration=True,
    ))
    
    # Connect controller
    builder.Connect(
        station.GetOutputPort("iiwa_state"),
        controller.GetInputPort("estimated_state"),
    )
    builder.Connect(
        motion_planner.GetOutputPort("iiwa_command"),
        controller.GetInputPort("desired_state"),
    )
    builder.Connect(
        motion_planner.GetOutputPort("iiwa_acceleration"),
        controller.GetInputPort("desired_acceleration"),
    )
    builder.Connect(
        controller.GetOutputPort("generalized_force"),
        station.GetInputPort("iiwa.actuation"),
    )
    
    return controller


# =============================================================================
# Simulation
# =============================================================================

def capture_object_point_cloud(plant, plant_context, point_cloud_system, 
                                simulator_context, obj_name, capture_position):
    """Move object to capture position and capture its point cloud."""
    obj = plant.GetModelInstanceByName(obj_name)
    body_idx = plant.GetBodyIndices(obj)[0]
    body = plant.get_body(body_idx)
    
    plant.SetFreeBodyPose(plant_context, body, RigidTransform(capture_position))
    point_cloud_system.CapturePointCloud(
        point_cloud_system.GetMyMutableContextFromRoot(simulator_context)
    )


def setup_throw(plant, plant_context, obj_name, throw_distance):
    """Configure object position and velocity for throwing."""
    # Determine object type
    obj_type = None
    for key in ['ball', 'banana', 'bottle']:
        if key in obj_name.lower():
            obj_type = key
            break
    
    if obj_type is None:
        print(f"Unknown object type: {obj_name}")
        return
    
    # Get rotation for this object and distance
    if throw_distance == "close":
        rotation = THROW_ROTATIONS_CLOSE[obj_type]
        throw_object_close(plant, plant_context, obj_name, rotation)
    else:
        rotation = THROW_ROTATIONS_FAR[obj_type]
        throw_object_far(plant, plant_context, obj_name, rotation)


def evaluate_catch(station, station_context, plant) -> bool:
    """Evaluate if the catch was successful."""
    body_poses = station.GetOutputPort("body_poses").Eval(station_context)
    
    # Get gripper pose
    gripper_idx = plant.GetBodyByName("body").index()
    gripper_pose = body_poses[gripper_idx]
    
    # Find object body
    obj_body_name = None
    for name in OBJECT_BODY_NAMES:
        if plant.HasBodyNamed(name):
            obj_body_name = name
            break
    
    if obj_body_name is None:
        print("Could not find object body")
        return False
    
    obj_idx = plant.GetBodyByName(obj_body_name).index()
    obj_pose = body_poses[obj_idx]
    
    distance, _ = calculate_obj_distance_to_gripper(gripper_pose, obj_pose)
    return distance < 0.05


def log_success(obj_name, grasp_seed):
    """Log successful catch to file."""
    with open('performance_test.txt', 'a') as f:
        f.write(f"{obj_name}\t{grasp_seed}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    print("Using main version 2")
    
    # Parse arguments
    args = parse_args()
    scenario_file, throw_distance, grasp_seed = get_scenario_config(args)
    
    # Constants
    RANDOM_SEED = 135
    POINT_CLOUD_CENTER = [0, 0, 100]
    SIMULATION_DURATION = 0.875
    
    np.random.seed(RANDOM_SEED)
    
    # Setup meshcat
    meshcat = StartMeshcat()
    meshcat.AddButton("Close")
    
    # Build diagram
    builder = DiagramBuilder()
    scenario = load_scenario(filename=scenario_file)
    
    # Hardware station
    station = builder.AddSystem(MakeHardwareStation(
        scenario=scenario,
        meshcat=meshcat,
        parser_preload_callback=lambda p: p.package_map().Add("cwd", os.getcwd()),
    ))
    scene_graph = station.GetSubsystemByName("scene_graph")
    plant = station.GetSubsystemByName("plant")
    
    # Find thrown object
    obj_name = find_thrown_object(plant)
    
    # Setup cameras
    icp_cameras, icp_transforms, pc_cameras, pc_transforms = setup_cameras(
        builder, station, plant, POINT_CLOUD_CENTER
    )
    
    # Setup perception
    point_cloud_system, traj_predictor = setup_perception(
        builder, icp_cameras, icp_transforms, pc_cameras, pc_transforms,
        POINT_CLOUD_CENTER, obj_name, plant, meshcat
    )
    
    # Setup planning
    grasp_selector, motion_planner = setup_planning(
        builder, station, plant, scene_graph, meshcat,
        traj_predictor, point_cloud_system, obj_name, grasp_seed
    )
    
    # Setup controller
    controller = setup_controller(builder, station, motion_planner)
    
    # Build and visualize diagram
    diagram = builder.Build()
    diagram.set_name("object_catching_system")
    diagram_visualize_connections(diagram, "diagram.svg")
    
    # Create simulator
    simulator = Simulator(diagram)
    simulator_context = simulator.get_mutable_context()
    station_context = station.GetMyMutableContextFromRoot(simulator_context)
    plant_context = plant.GetMyMutableContextFromRoot(simulator_context)
    
    # Capture object point cloud
    capture_object_point_cloud(
        plant, plant_context, point_cloud_system,
        simulator_context, obj_name, POINT_CLOUD_CENTER
    )
    
    # Setup throw
    setup_throw(plant, plant_context, obj_name, throw_distance)
    
    # Run simulation
    simulator.set_target_realtime_rate(1)
    simulator.set_publish_every_time_step(True)
    plt.show()
    
    meshcat.StartRecording()
    simulator.AdvanceTo(SIMULATION_DURATION)
    meshcat.PublishRecording()
    
    # Evaluate result
    if evaluate_catch(station, station_context, plant):
        print("CATCH SUCCESS")
        log_success(obj_name, grasp_seed)
    
    # Wait for user to close
    while not meshcat.GetButtonClicks("Close"):
        pass


if __name__ == "__main__":
    main()