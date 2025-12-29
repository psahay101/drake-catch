# Drake Catch 🤖🎾

Real-time robotic catching system using Drake simulation. An iiwa robot arm tracks, predicts, and catches thrown objects mid-air.


## Overview

This project demonstrates:
- **Multi-camera perception** for real-time object tracking
- **Trajectory prediction** using ICP and RANSAC
- **Grasp selection** using Darboux frame analysis
- **Real-time motion planning** with B-spline trajectory optimization

## How It Works

```
Cameras → Point Cloud → ICP Tracking → Trajectory Prediction
                                              ↓
                              Grasp Selection (Darboux frames)
                                              ↓
                              Motion Planning (KinematicTrajectoryOptimization)
                                              ↓
                              Inverse Dynamics Control → Robot Catches Object
```

## Installation

### Prerequisites
- Python 3.10+
- [Drake](https://drake.mit.edu/) (tested with v1.x)
- [Manipulation](https://github.com/RussTedrake/manipulation) library

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/drake-catch.git
cd drake-catch

# Install dependencies (assuming Drake is already installed)
pip install numpy scipy matplotlib
```

## Usage

```bash
# Catch a tennis ball (close throw)
python src/main.py --obj t --distance c --randomization 0

# Catch a banana (far throw)
python src/main.py --obj b --distance f --randomization 42

# Catch a pill bottle
python src/main.py --obj p --distance c --randomization 123
```

### Arguments

| Argument | Options | Description |
|----------|---------|-------------|
| `--obj` | `t`, `b`, `p` | Object: tennis ball, banana, pill bottle |
| `--distance` | `c`, `f` | Throw distance: close, far |
| `--randomization` | integer | Random seed for grasp sampling |

## Project Structure

```
drake-catch/
├── src/
│   ├── main.py              # Entry point
│   ├── perception.py        # Camera, point cloud, trajectory prediction
│   ├── grasping_selection.py # Darboux frame grasp sampling
│   ├── motion_planner.py    # B-spline trajectory optimization
│   ├── station.py           # Drake hardware station config
│   └── utils.py             # Helper functions
├── data/
│   ├── *.sdf                # Object models
│   ├── *.obj / *.mtl        # Mesh files
│   └── scenario_*.yaml      # Simulation scenarios
└── README.md
```

## Key Components

### Perception (`perception.py`)
- Spherical camera array for multi-view depth sensing
- ICP (Iterative Closest Point) for pose estimation
- RANSAC for robust trajectory fitting under ballistic motion model

### Grasp Selection (`grasping_selection.py`)
- Darboux frame computation on object surface
- Cost function considering:
  - Gripper-to-centroid alignment
  - Gripper orientation relative to robot base
  - Alignment with object velocity vector
- Parallel collision checking

### Motion Planning (`motion_planner.py`)
- `KinematicTrajectoryOptimization` with B-spline trajectories
- Iterative constraint tightening for warm-starting
- Position, orientation, and velocity constraints at catch point
- 25ms replanning loop

## Results

The system successfully catches objects with varying:
- Shapes (spherical, elongated, cylindrical)
- Trajectories (close/far throws)
- Randomized grasp sampling

## References

- [Drake](https://drake.mit.edu/) - Simulation and optimization
- [Darboux Frame Grasping](https://arxiv.org/pdf/1706.09911.pdf) - Grasp sampling method
- [Underactuated Robotics](https://underactuated.mit.edu/) - Motion planning concepts

## License

MIT License

## Acknowledgments

Built using Drake and the Manipulation library from MIT.
