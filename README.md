# r2d2 — ROS 2 package

This package contains the R2-D2 robot’s ROS 2 workspace components: build and package metadata, configuration, robot description, launch files, and simulation worlds.

## Repository layout

```
dev_ws/src/r2d2/
├── CMakeLists.txt
├── package.xml
├── config/
├── description/
├── launch/
└── worlds/
```

### Top-level files

- CMakeLists.txt  
  Colcon/CMake build script for the package. Defines targets, dependencies, install rules, and export settings used by the ROS 2 build system.

- package.xml  
  ROS 2 package manifest. Declares the package’s name, version, description, maintainers, licenses, and build/run/test dependencies. Tools like `rosdep` and `colcon` use this to resolve and validate dependencies.

### Directories

- config/  
  Configuration assets loaded at runtime. Typical contents include:
  - ROS 2 node parameter YAML files (e.g., navigation, sensors, controllers).
  - Controller Manager and joint/controller configuration (if using ros2_control).
  - RViz or teleop configuration files.

  How it’s used: launch files generally pass these YAMLs to nodes via the `parameters` argument, or they’re referenced by nodes reading config paths.

- description/  
  Robot model and assets. Typical contents include:
  - URDF/Xacro robot description files defining links, joints, sensors, and plugins.
  - Meshes (STL/DAE) and materials for visualization and simulation.
  - ROS 2 control/transmission tags if actuated joints are present.

  How it’s used: the robot description is published to the `robot_description` parameter, used by RViz, planning frameworks, and simulators.

- launch/  
  ROS 2 Python launch files orchestrating one or more nodes. Typical contents include:
  - Bringup files that start the robot state publisher, controllers, and visualization.
  - Simulation bringup that loads the robot into a Gazebo world alongside sensors and plugins.
  - Launch arguments for selecting configs, worlds, and headless/headful modes.

  How it’s used: run with `ros2 launch r2d2 <file>.launch.py` (replace with the actual launch file names found in this directory).

- worlds/  
  Simulation worlds (e.g., for Gazebo). Typical contents include:
  - SDF/World files describing environments, lights, physics, and assets.
  - Auxiliary models referenced by the world files.

  How it’s used: launch files typically load a chosen world and spawn the robot model into it.

## Build and workspace

This package is part of a colcon workspace under `dev_ws`. A common workflow is:

- Source your ROS 2 distribution setup.
- From `dev_ws`, build with `colcon build`.
- Source the local setup: `source install/setup.bash`.

## Conventions

- File naming: prefer lower_snake_case for YAML and launch files; use `.xacro` for parameterized robot models.
- Paths: use package-relative paths in launch files (via `ament_index_python` / `get_package_share_directory`) to keep files relocatable.
- Parameters: keep node parameters in `config/` and pass them via launch, rather than hardcoding.

## Notes

- Update `package.xml` when adding new dependencies required by launch files, URDF plugins, or tools.
- Ensure install rules in `CMakeLists.txt` export the package share contents (config, description, launch, worlds) so downstream packages and launch files can locate them.
