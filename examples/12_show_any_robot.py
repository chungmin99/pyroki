# Minimal PyRoki + Viser humanoid visualization
import pyroki as pk
import viser
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description
import numpy as np
import time

# Minimal PyRoki + Viser example. 

# Visualize any robot from the list of available robot descriptions
ROBOT_DESCRIPTIONS = [
    'a1_description', 'ability_hand_description', 'aliengo_description', 'allegro_hand_description',
    'anymal_b_description', 'anymal_c_description', 'anymal_d_description', 'atlas_drc_description',
    'atlas_v4_description', 'b1_description', 'b2_description', 'baxter_description',
    'berkeley_humanoid_description', 'bolt_description', 'cassie_description', 'cf2_description',
    'double_pendulum_description', 'draco3_description', 'elf2_description', 'ergocub_description',
    'fanuc_m710ic_description', 'fetch_description', 'finger_edu_description', 'g1_description',
    'gen2_description', 'gen3_description', 'gen3_lite_description', 'go1_description', 'go2_description',
    'h1_description', 'hyq_description', 'icub_description', 'iiwa14_description', 'iiwa7_description',
    'jaxon_description', 'jvrc_description', 'nextage_description', 'panda_description', 'piper_description',
    'poppy_torso_description', 'pr2_description', 'r2_description', 'reachy_description', 'rhea_description',
    'robotiq_2f85_description', 'rsk_description', 'solo_description', 'spryped_description', 'talos_description',
    'tiago_description', 'trifinger_edu_description', 'ur10_description', 'ur3_description', 'ur5_description',
    'yumi_description', 'z1_description'
]


def main():
    robot_name = "a1_description"
    try:
        urdf = load_robot_description(robot_name)
        robot = pk.Robot.from_urdf(urdf)
    except Exception as e:
        print(f"Failed to load {robot_name}: {e}")

    server = viser.ViserServer()
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    server.scene.add_grid("/ground", width=2, height=2)

    # Visualize the default configuration
    urdf_vis.update_cfg(np.zeros(len(robot.joints.names)))
    # Keep the server running
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
