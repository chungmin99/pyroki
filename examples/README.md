# Custom URDF IK with PyRoki and Viser

This script (13_custom_urdf_ik.py) demonstrates how to control a custom robot using inverse kinematics (IK) with `pyroki`, `viser`, and `yourdfpy`. 
It allows you to load a robot from a URDF file or use a pre-defined robot from `robot_descriptions` and control its end-effector or user defined targer link.

## Features

* Load custom URDF files or use built-in robot descriptions.
* Inverse Kinematics (IK) solving to control the robot's end-effector or user defined targer link.
* Visualization of the robot using Viser.

## Limitations

* Capsule shape is not a standard primitive geometry type directly supported by URDF. URDF primarily supports `<box>`, `<cylinder>`, and `<sphere>`.
* Continuous joints are not yet fully supported.
* If the URDF contains mesh, DAE, or STL files, the path should be specified correctly.
* Paths like `<mesh filename="package://visual/base_link.dae"/>` might not work as `yourdfpy` interprets "visual" as the package name. 
* Ensure your mesh paths are correctly structured, e.g., `<mesh filename="package://your_robot_package_name/visual/base_link.dae"/>`
* For example: `<mesh filename="package://kuka_kr3_support/meshes/kr3r540/visual/base_link.stl"/>`

## Usage

To run the script, you can use the following command-line arguments:

```bash
python 13_custom_urdf_ik.py [--robot-type <robot_type>] [--urdf-path <path_to_custom_urdf>] [--target-link-name <link_name>]

python 13_custom_urdf_ik.py --robot-type ur10_description
python 13_custom_urdf_ik.py --urdf-path D:\Python_projects\PyRokiControl\custom_urdf\01_custom_arm.urdf --target-link-name end_effector_tool

more urdf examples at: https://github.com/Daniella1/urdf_files_dataset/tree/main/urdf_files/
