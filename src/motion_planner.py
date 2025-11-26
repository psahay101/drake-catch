import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    LeafSystem,
    AbstractValue,
    DiagramBuilder,
    BsplineTrajectory,
    CompositeTrajectory,
    PiecewisePolynomial,
    PathParameterizedTrajectory,
    KinematicTrajectoryOptimization,
    Parser,
    PositionConstraint,
    OrientationConstraint,
    SpatialVelocityConstraint,
    RigidTransform,
    Solve,
    RotationMatrix,
    JacobianWrtVariable,
    MultibodyPlant,
)

from manipulation.meshcat_utils import AddMeshcatTriad
from pydrake.multibody import inverse_kinematics
from pydrake.solvers import SnoptSolver

from utils import ObjectTrajectory, calculate_obj_distance_to_gripper


@dataclass
class TrajInputPack:
    """All inputs needed for trajectory planning."""
    obj_traj: ObjectTrajectory
    current_gripper_pose: RigidTransform
    q_current: np.ndarray  # (7,) joint positions
    iiwa_vels: np.ndarray  # (7,) joint velocities
    grasp_pose: RigidTransform
    obj_catch_time: float
    current_time: float


@dataclass 
class ConstraintParams:
    """Parameters for trajectory optimization constraints."""
    duration_err: float = 0.01
    pos_err: float = 0.02
    theta_bound: float = 0.1
    vel_err: float = 0.4
    
    def tighten(self, factor: float = 0.875) -> 'ConstraintParams':
        """Return new params tightened by factor."""
        return ConstraintParams(
            duration_err=self.duration_err * factor,
            pos_err=self.pos_err * factor,
            theta_bound=self.theta_bound * factor,
            vel_err=self.vel_err * factor,
        )


class MotionPlanner(LeafSystem):
    """
    Constrained trajectory optimization for iiwa to intercept a moving object.
    
    Uses KinematicTrajectoryOptimization with B-spline trajectories, iteratively
    tightening constraints from a warm start.
    """
    
    # Transform from iiwa_link_7 to gripper frame
    LINK7_TO_GRIPPER = RotationMatrix.MakeZRotation(np.pi / 2) @ RotationMatrix.MakeXRotation(np.pi / 2)
    GRIPPER_OFFSET = np.array([0, 0, 0.1])  # offset from link_7 to gripper tip
    
    # Optimization parameters
    NUM_CONTROL_POINTS = 8
    MAX_ITERATIONS = 12
    UPDATE_PERIOD = 0.025  # seconds
    MIN_TIME_TO_CATCH = 0.2  # stop updating when closer than this to catch
    PRE_CATCH_OFFSET = 0.6  # time offset for pre-catch waypoint
    CATCH_VEL_SCALE = 0.3  # fraction of object velocity to match at catch
    
    def __init__(self, original_plant: MultibodyPlant, meshcat):
        LeafSystem.__init__(self)
        
        print("Using motion planning version 2")
        
        self.original_plant = original_plant
        self.meshcat = meshcat
        self.visualize = True
        
        # Nominal joint config for regularization
        self.q_nominal = np.array([0.0, 0.6, 0.0, -1.75, 0.0, 1.0, 0.0])
        
        # State tracking
        self.previous_traj: Optional[BsplineTrajectory] = None
        self.obj_vel_at_catch: Optional[np.ndarray] = None
        self.desired_wsg_state = 1  # 1=open, 0=closed
        
        # Cache plants to avoid rebuilding every cycle
        self._cached_plant: Optional[Tuple[MultibodyPlant, int]] = None
        
        self._setup_ports()
        self.DeclarePeriodicUnrestrictedUpdateEvent(self.UPDATE_PERIOD, 0.0, self.compute_traj)

    def _setup_ports(self):
        """Declare all input/output ports and state."""
        # Input ports
        grasp = AbstractValue.Make({RigidTransform(): 0})
        self.DeclareAbstractInputPort("grasp_selection", grasp)
        
        body_poses = AbstractValue.Make([RigidTransform()])
        self.DeclareAbstractInputPort("iiwa_current_pose", body_poses)
        
        obj_traj = AbstractValue.Make(ObjectTrajectory())
        self.DeclareAbstractInputPort("object_trajectory", obj_traj)
        
        self.DeclareVectorInputPort(name="iiwa_state", size=14)
        
        # State for storing computed trajectory
        default_traj = CompositeTrajectory([
            PiecewisePolynomial.FirstOrderHold([0, 1], np.array([[0, 0]]))
        ])
        self._traj_index = self.DeclareAbstractState(AbstractValue.Make(default_traj))
        
        # Output ports
        self.DeclareVectorOutputPort("iiwa_command", 14, self.output_traj)
        self.DeclareVectorOutputPort("iiwa_acceleration", 7, self.output_acceleration)
        self.DeclareVectorOutputPort("wsg_command", 1, self.output_wsg_traj)

    def _get_plant(self) -> Tuple[MultibodyPlant, int]:
        """Get or create cached iiwa plant."""
        if self._cached_plant is None:
            builder = DiagramBuilder()
            plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
            iiwa_model = Parser(plant).AddModelsFromUrl(
                "package://drake/manipulation/models/iiwa_description/urdf/"
                "iiwa14_spheres_dense_collision.urdf"
            )[0]
            plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("base"))
            plant.Finalize()
            self._cached_plant = (plant, iiwa_model)
        return self._cached_plant

    def _load_inputs(self, context) -> Optional[TrajInputPack]:
        """Load and validate all trajectory planning inputs."""
        # Object trajectory
        obj_traj = self.get_input_port(2).Eval(context)
        if obj_traj == ObjectTrajectory():
            return None
        
        # Current gripper pose
        body_poses = self.get_input_port(1).Eval(context)
        gripper_body_idx = self.original_plant.GetBodyByName("body").index()
        current_gripper_pose = body_poses[gripper_body_idx]
        
        # Current iiwa state
        iiwa_state = self.get_input_port(3).Eval(context)
        q_current = iiwa_state[:7]
        iiwa_vels = iiwa_state[7:]
        
        # Grasp selection
        grasp = self.get_input_port(0).Eval(context)
        grasp_pose = list(grasp.keys())[0]
        obj_catch_time = list(grasp.values())[0]
        
        if grasp_pose.IsExactlyEqualTo(RigidTransform()):
            return None
        
        # Too close to catch time
        if obj_catch_time - context.get_time() < self.MIN_TIME_TO_CATCH:
            return None
        
        return TrajInputPack(
            obj_traj=obj_traj,
            current_gripper_pose=current_gripper_pose,
            q_current=q_current,
            iiwa_vels=iiwa_vels,
            grasp_pose=grasp_pose,
            obj_catch_time=obj_catch_time,
            current_time=context.get_time(),
        )

    def _compute_gripper_velocity(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """Compute gripper Cartesian velocity from joint state via Jacobian."""
        plant, model = self._get_plant()
        context = plant.CreateDefaultContext()
        plant.SetPositions(context, model, q)
        
        J = plant.CalcJacobianTranslationalVelocity(
            context,
            JacobianWrtVariable.kQDot,
            plant.GetFrameByName("iiwa_link_7"),
            self.GRIPPER_OFFSET,
            plant.world_frame(),
            plant.world_frame(),
        )
        return J @ qdot

    def _solve_ik(self, target_pose: RigidTransform, 
                  use_gripper_transform: bool = True) -> Optional[np.ndarray]:
        """
        Solve inverse kinematics for target pose.
        
        Args:
            target_pose: Desired gripper pose in world frame
            use_gripper_transform: If True, account for link7-to-gripper transform
            
        Returns:
            Joint angles or None if failed
        """
        plant, _ = self._get_plant()
        world_frame = plant.world_frame()
        gripper_frame = plant.GetFrameByName("iiwa_link_7")
        
        ik = inverse_kinematics.InverseKinematics(plant)
        q_vars = ik.q()
        prog = ik.prog()
        
        # Regularization toward nominal
        prog.AddQuadraticErrorCost(np.eye(len(q_vars)), self.q_nominal, q_vars)
        
        # Position constraint
        ik.AddPositionConstraint(
            frameA=world_frame,
            frameB=gripper_frame,
            p_BQ=self.GRIPPER_OFFSET,
            p_AQ_lower=target_pose.translation(),
            p_AQ_upper=target_pose.translation(),
        )
        
        # Orientation constraint
        R_BbarB = self.LINK7_TO_GRIPPER if use_gripper_transform else RotationMatrix()
        ik.AddOrientationConstraint(
            frameAbar=world_frame,
            R_AbarA=target_pose.rotation(),
            frameBbar=gripper_frame,
            R_BbarB=R_BbarB,
            theta_bound=0.05,
        )
        
        prog.SetInitialGuess(q_vars, self.q_nominal)
        result = Solve(prog)
        
        if not result.is_success():
            print(f"IK failed: {result.get_solver_id().name()}")
            print(result.GetInfeasibleConstraintNames(prog))
            return None
        
        return result.GetSolution(q_vars)

    def _configure_solver(self, prog):
        """Set SNOPT solver tolerances."""
        solver_id = SnoptSolver().solver_id()
        for tol_name in ["Feasibility tolerance", "Major feasibility tolerance",
                         "Minor feasibility tolerance", "Major optimality tolerance",
                         "Minor optimality tolerance"]:
            prog.SetSolverOption(solver_id, tol_name, 0.001)

    def _add_trajopt_constraints(
        self,
        trajopt: KinematicTrajectoryOptimization,
        plant: MultibodyPlant,
        plant_autodiff,
        inputs: TrajInputPack,
        current_gripper_vel: np.ndarray,
        params: ConstraintParams,
    ):
        """Add all constraints to trajectory optimization problem."""
        plant_context = plant.CreateDefaultContext()
        world_frame = plant.world_frame()
        gripper_frame = plant.GetFrameByName("iiwa_link_7")
        
        # Basic bounds
        trajopt.AddPositionBounds(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())
        trajopt.AddVelocityBounds(plant.GetVelocityLowerLimits(), plant.GetVelocityUpperLimits())
        
        # Compute timing
        time_to_catch = inputs.obj_catch_time - inputs.current_time
        duration = time_to_catch + self.PRE_CATCH_OFFSET
        catch_normalized = time_to_catch / duration
        
        pre_catch_time = time_to_catch - 0.12
        pre_catch_normalized = pre_catch_time / duration
        
        trajopt.AddDurationConstraint(duration - params.duration_err, duration + params.duration_err)
        
        # Compute pre-catch pose (where gripper should be 0.12s before catch)
        X_WO = inputs.obj_traj.value(inputs.obj_catch_time)
        X_OG_W = X_WO.inverse() @ inputs.grasp_pose
        X_WPreGoal = inputs.obj_traj.value(inputs.obj_catch_time - 0.12) @ X_OG_W
        
        # Store constraints to prevent deallocation (Drake binding issue)
        self._active_constraints = []
        
        def add_pose_constraint(pose: RigidTransform, t_normalized: float, 
                                pos_tol: float, name: str):
            """Helper to add position + orientation constraint at normalized time."""
            pos_constraint = PositionConstraint(
                plant, world_frame,
                pose.translation() - pos_tol,
                pose.translation() + pos_tol,
                gripper_frame, self.GRIPPER_OFFSET,
                plant_context,
            )
            orient_constraint = OrientationConstraint(
                plant, world_frame, pose.rotation(),
                gripper_frame, self.LINK7_TO_GRIPPER,
                params.theta_bound, plant_context,
            )
            trajopt.AddPathPositionConstraint(pos_constraint, t_normalized)
            trajopt.AddPathPositionConstraint(orient_constraint, t_normalized)
            self._active_constraints.extend([pos_constraint, orient_constraint])
        
        # Start constraint
        add_pose_constraint(inputs.current_gripper_pose, 0.0, params.pos_err, "start")
        
        # Pre-catch constraint (looser)
        pre_catch_tol = max(params.pos_err * 25, 0.2)
        add_pose_constraint(X_WPreGoal, pre_catch_normalized, pre_catch_tol, "pre_catch")
        
        # Catch constraint
        add_pose_constraint(inputs.grasp_pose, catch_normalized, params.pos_err, "catch")
        
        # Velocity constraints
        autodiff_context = plant_autodiff.CreateDefaultContext()
        gripper_frame_autodiff = plant_autodiff.GetFrameByName("iiwa_link_7")
        
        start_vel_constraint = SpatialVelocityConstraint(
            plant_autodiff,
            plant_autodiff.world_frame(),
            current_gripper_vel - params.vel_err,
            current_gripper_vel + params.vel_err,
            gripper_frame_autodiff,
            self.GRIPPER_OFFSET.reshape(-1, 1),
            autodiff_context,
        )
        
        obj_vel = inputs.obj_traj.EvalDerivative(inputs.obj_catch_time)[:3]
        catch_vel = obj_vel * self.CATCH_VEL_SCALE
        
        final_vel_constraint = SpatialVelocityConstraint(
            plant_autodiff,
            plant_autodiff.world_frame(),
            catch_vel - params.vel_err,
            catch_vel + params.vel_err,
            gripper_frame_autodiff,
            self.GRIPPER_OFFSET.reshape(-1, 1),
            autodiff_context,
        )
        
        trajopt.AddVelocityConstraintAtNormalizedTime(start_vel_constraint, 0)
        trajopt.AddVelocityConstraintAtNormalizedTime(final_vel_constraint, catch_normalized)
        self._active_constraints.extend([start_vel_constraint, final_vel_constraint])

    def _solve_trajopt(
        self,
        inputs: TrajInputPack,
        current_gripper_vel: np.ndarray,
        initial_guess: Optional[BsplineTrajectory],
        params: ConstraintParams,
    ) -> Optional[BsplineTrajectory]:
        """Run single trajectory optimization solve."""
        traj, _ = self._solve_trajopt_with_result(inputs, current_gripper_vel, initial_guess, params)
        return traj

    def _solve_trajopt_with_result(
        self,
        inputs: TrajInputPack,
        current_gripper_vel: np.ndarray,
        initial_guess: Optional[BsplineTrajectory],
        params: ConstraintParams,
    ) -> Tuple[Optional[BsplineTrajectory], Optional[BsplineTrajectory]]:
        """
        Run single trajectory optimization solve.
        
        Returns:
            Tuple of (success_traj, fallback_traj) where:
            - success_traj: The trajectory if solve succeeded, else None
            - fallback_traj: The reconstructed trajectory even if solve failed (for warm starting)
        """
        plant, _ = self._get_plant()
        plant_autodiff = plant.ToAutoDiffXd()
        num_q = plant.num_positions()
        
        trajopt = KinematicTrajectoryOptimization(num_q, self.NUM_CONTROL_POINTS)
        prog = trajopt.get_mutable_prog()
        self._configure_solver(prog)
        
        self._add_trajopt_constraints(
            trajopt, plant, plant_autodiff, inputs, current_gripper_vel, params
        )
        
        # Set initial guess
        if initial_guess is not None:
            trajopt.SetInitialGuess(initial_guess)
        else:
            # Use IK solution for endpoint, linearly interpolate
            # Note: use_gripper_transform=False to match original behavior
            # The IK guess uses identity rotation, trajopt constraints use the actual transform
            q_end = self._solve_ik(inputs.grasp_pose, use_gripper_transform=False)
            if q_end is None:
                q_end = self.q_nominal
            q_guess = np.linspace(inputs.q_current, q_end, self.NUM_CONTROL_POINTS).T
            trajopt.SetInitialGuess(BsplineTrajectory(trajopt.basis(), q_guess))
        
        result = SnoptSolver().Solve(prog)
        fallback_traj = trajopt.ReconstructTrajectory(result)
        
        if not result.is_success():
            print(f"Trajopt failed: {result.get_solver_id().name()}")
            print(result.GetInfeasibleConstraintNames(prog))
            return None, fallback_traj
        
        return fallback_traj, fallback_traj

    def _visualize_trajectory(self, traj, name: str):
        """Draw trajectory path in meshcat."""
        if not self.visualize:
            return
            
        plant, model = self._get_plant()
        context = plant.CreateDefaultContext()
        
        NUM_STEPS = 50
        positions = np.zeros((3, NUM_STEPS))
        
        for i, t in enumerate(np.linspace(traj.start_time(), traj.end_time(), NUM_STEPS)):
            plant.SetPositions(context, model, traj.value(t))
            pos = plant.CalcRelativeTransform(
                context, plant.world_frame(), plant.GetFrameByName("iiwa_link_7")
            ).translation()
            positions[:, i] = pos
        
        self.meshcat.SetLine(name, positions)

    def _shift_trajectory_time(self, traj: BsplineTrajectory, 
                                start_time: float) -> PathParameterizedTrajectory:
        """Shift trajectory to start at given time."""
        duration = traj.end_time() - traj.start_time()
        time_scaling = PiecewisePolynomial.FirstOrderHold(
            [start_time, start_time + duration],
            np.array([[0, duration]])
        )
        return PathParameterizedTrajectory(traj, time_scaling)

    def compute_traj(self, context, state):
        """Periodic update: compute new catching trajectory."""
        print("motion_planner update event")
        
        inputs = self._load_inputs(context)
        if inputs is None:
            return
        
        current_gripper_vel = self._compute_gripper_velocity(inputs.q_current, inputs.iiwa_vels)
        obj_vel = inputs.obj_traj.EvalDerivative(inputs.obj_catch_time)[:3]
        print(f"current_gripper_vel: {current_gripper_vel}")
        print(f"obj_vel_at_catch: {obj_vel}")
        
        # Visualize goal
        if self.visualize:
            AddMeshcatTriad(self.meshcat, "goal", X_PT=inputs.grasp_pose, opacity=0.5)
            self.meshcat.SetTransform("goal", inputs.grasp_pose)
        
        # First time: iterative solve with progressively tighter constraints
        if self.previous_traj is None:
            final_traj = self._iterative_solve(inputs, current_gripper_vel)
        else:
            # Subsequent: single solve with tight constraints using previous as guess
            self._clear_iteration_visuals()
            params = ConstraintParams()  # default tight params
            final_traj = self._solve_trajopt(inputs, current_gripper_vel, self.previous_traj, params)
            
            if final_traj is None:
                print("Tight solve failed, keeping previous trajectory")
                return
            print("Tight solve succeeded")
        
        if final_traj is None:
            return
        
        # Shift to current time and store
        shifted_traj = self._shift_trajectory_time(final_traj, inputs.current_time)
        self._visualize_trajectory(shifted_traj, "final traj")
        
        state.get_mutable_abstract_state(int(self._traj_index)).set_value(shifted_traj)
        self.previous_traj = final_traj
        self.obj_vel_at_catch = obj_vel

    def _iterative_solve(self, inputs: TrajInputPack, 
                         current_gripper_vel: np.ndarray) -> Optional[BsplineTrajectory]:
        """Iteratively solve with progressively tighter constraints."""
        params = ConstraintParams(
            duration_err=0.05,
            pos_err=0.1,
            theta_bound=0.8,
            vel_err=2.0,
        )
        
        final_traj = None
        
        for i in range(self.MAX_ITERATIONS):
            traj, trajopt_result = self._solve_trajopt_with_result(
                inputs, current_gripper_vel, final_traj, params
            )
            
            if traj is None:
                print(f"Iteration {i} failed")
                # On first iteration failure, still try to get a trajectory as fallback
                if final_traj is None and trajopt_result is not None:
                    final_traj = trajopt_result
                    print("Using failed solve result as fallback")
                break
            
            print(f"Iteration {i} succeeded")
            final_traj = traj
            
            # Visualize this iteration
            shifted = self._shift_trajectory_time(traj, inputs.current_time)
            self._visualize_trajectory(shifted, f"traj iter={i}")
            
            params = params.tighten()
        
        return final_traj

    def _clear_iteration_visuals(self):
        """Remove iteration trajectory visuals from meshcat."""
        if not self.visualize:
            return
        for i in range(self.MAX_ITERATIONS):
            try:
                self.meshcat.Delete(f"traj iter={i}")
            except:
                pass

    def _get_current_traj(self, context):
        """Get trajectory from state, return None if default/invalid."""
        traj = context.get_mutable_abstract_state(int(self._traj_index)).get_value()
        return None if traj.rows() == 1 else traj

    def _get_default_position(self) -> np.ndarray:
        """Get default iiwa position."""
        default_context = self.original_plant.CreateDefaultContext()
        iiwa_model = self.original_plant.GetModelInstanceByName("iiwa")
        return self.original_plant.GetPositions(default_context, iiwa_model)

    def output_traj(self, context, output):
        """Output current trajectory position and velocity."""
        traj = self._get_current_traj(context)
        
        if traj is None:
            output.SetFromVector(np.concatenate([self._get_default_position(), np.zeros(7)]))
        else:
            t = context.get_time()
            output.SetFromVector(np.concatenate([traj.value(t).flatten(), 
                                                  traj.EvalDerivative(t).flatten()]))

    def output_acceleration(self, context, output):
        """Output current trajectory acceleration."""
        traj = self._get_current_traj(context)
        
        if traj is None:
            output.SetFromVector(np.zeros(7))
        else:
            output.SetFromVector(traj.EvalDerivative(context.get_time(), 2).flatten())

    def output_wsg_traj(self, context, output):
        """Output gripper command based on object proximity."""
        body_poses = self.get_input_port(1).Eval(context)
        
        gripper_body_idx = self.original_plant.GetBodyByName("body").index()
        current_gripper_pose = body_poses[gripper_body_idx]
        
        # Find object body
        obj_name = None
        for name in ["Tennis_ball", "Banana", "pill_bottle"]:
            if self.original_plant.HasBodyNamed(name):
                obj_name = name
                break
        
        if obj_name is None:
            output.SetFromVector(np.array([1]))  # default open
            return
        
        obj_body_idx = self.original_plant.GetBodyByName(obj_name).index()
        current_obj_pose = body_poses[obj_body_idx]
        
        distance, vec_to_obj = calculate_obj_distance_to_gripper(current_gripper_pose, current_obj_pose)
        
        DISTANCE_THRESHOLD = 0.01
        ROUGH_RANGE = 0.25
        
        # Close gripper if object is close enough, and keep closed once triggered
        if (np.linalg.norm(vec_to_obj) < ROUGH_RANGE and distance < DISTANCE_THRESHOLD) or self.desired_wsg_state == 0:
            self.desired_wsg_state = 0
            output.SetFromVector(np.array([0]))
        else:
            output.SetFromVector(np.array([1]))