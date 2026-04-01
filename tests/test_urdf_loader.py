"""Basic IK

Simple test to verify pyroki and some of its dependencies are properly installed
"""

import time

import numpy as np
import pyroki as pk
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf


def test_urdf_loader():
    """Main function for basic IK."""

    urdf = load_robot_description("panda_description")
    target_link_name = "panda_hand"

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)

if __name__ == "__main__":
    test_urdf_loader()
