import numpy as np
import jax
import jax.numpy as jnp
from collimator import library
from collimator.backend import io_callback
import argparse
from src.util import SavePlot, RenderVideo, ConfigureMujoco
import mujoco
from src.system import RunSystem, BuildSystem


def main():
        # Add parser argument to upload xml file
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_file", default= "./resources/bi-franka/scene.xml", help="scene file")
    args = parser.parse_args()
    
    
    model = mujoco.MjModel.from_xml_path(args.xml_file)


    left_hand_id = model.body("fa-hand").id  # The end effector ID
    right_hand_id = model.body("fb-hand").id  # The end effector ID

    left_target_id = model.site("target_left").id  # The block ("target") ID
    right_target_id = model.site("target_right").id  # The block ("target") ID

    print(f'Right target: {right_target_id}, Left target: {left_target_id}')


    left_arm_actuator_names = [
        "fa-actuator1",
        "fa-actuator2",
        "fa-actuator3",
        "fa-actuator4",
        "fa-actuator5",
        "fa-actuator6",
        "fa-actuator7",
    ]

    right_arm_actuator_names = [
        "fb-actuator2",
        "fb-actuator3",
        "fb-actuator4",
        "fb-actuator5",
        "fb-actuator6",
        "fb-actuator7",
        "fb-actuator8"
    ]


    # Describes the ids for the left and right arms
    left_arm_actuator_ids = np.array([model.actuator(name).id for name in left_arm_actuator_names])
    right_arm_actuator_ids = np.array([model.actuator(name).id for name in right_arm_actuator_names])
    print(right_arm_actuator_ids)
    
    dt = 0.001

    xml_file= "./resources/bi-franka/scene.xml"
    
    data, mjmodel = ConfigureMujoco(
        model,
        dt,
        xml_file,
        left_hand_id,
        right_hand_id,
        left_target_id,
        right_target_id
    )
    
    
    
    # Define key locations in the scene
    left_home_pos = data.body(left_hand_id).xpos.copy()
    left_u0 = data.ctrl[left_arm_actuator_ids].copy()
    left_target_pos = data.site(left_target_id).xpos.copy()
    left_pregrasp_pos = left_target_pos + np.array([-0.01, 0.0, 0.35])
    left_grasp_pos = left_target_pos + np.array([-0.01, 0.0, 0.05])
    left_waypoint_pos = left_home_pos + np.array([0.0, 0.0, -0.1])
    left_preplace_pos = np.array([0.6, 0.35, 0.5])
    left_place_pos = left_preplace_pos + np.array([0.0, 0.0, -0.12])

    right_home_pos = data.body(right_hand_id).xpos.copy()
    right_u0 = data.ctrl[right_arm_actuator_ids].copy()
    right_target_pos = data.site(right_target_id).xpos.copy()
    right_pregrasp_pos = right_target_pos + np.array([-0.01, 0.0, 0.35])
    right_grasp_pos = right_target_pos + np.array([.01, 0.0, 0.04])
    right_waypoint_pos = right_home_pos + np.array([0.0, 0.0, -0.1])
    right_preplace_pos = np.array([0.6, 0.35, 0.5])
    right_place_pos = right_preplace_pos + np.array([0.0, 0.0, -0.12])
    
        # Custom block to output piecewise linear target locations
    OPEN = np.array(0)
    CLOSE = np.array(1)

    # Motion stages (times are cumulative in seconds)
    scale = 2.0
    t0 = 0.0
    left_time_array = np.array([
        0.0,  # Home
        2.0,  # Pre-grasp
        8.0,  # Grasp
        8.5,  # Close gripper (grasp)
        8.7,  # Pre-grasp
        15.0,  # Waypoint
    ]) / scale

    tf = left_time_array[-1]

    left_xpos_array = np.array([
        left_home_pos,
        left_pregrasp_pos,
        left_grasp_pos,
        left_grasp_pos,
        left_pregrasp_pos,
        left_waypoint_pos,
    ])

    right_time_array = np.array([
        0.0,  # Home
        2.0,  # Pre-grasp
        8.0,  # Grasp
        8.5,  # Close gripper (grasp)
        8.7,  # Pre-grasp
        15.0,  # Waypoint
    ]) / scale
    tf_right = right_time_array[-1]

    right_xpos_array = np.array([
        right_home_pos,
        right_pregrasp_pos,
        right_grasp_pos,
        right_grasp_pos,
        right_pregrasp_pos,
        right_waypoint_pos,
    ])

    # vmap interpolation to allow for 2d array input
    interp_fun = jax.vmap(jnp.interp, (None, None, 1))

    def left_pos_command_cb(time):
        t = time - t0
        return jnp.where((t < 0.0) | (t > tf), left_home_pos, interp_fun(t, left_time_array, left_xpos_array))

    def left_grip_command_cb(time):
        t = time - t0
        return jnp.where((t > left_time_array[2]) & (t < left_time_array[5]), 0.0, 255.0)

    def right_pos_command_cb(time):
        t = time - t0
        return jnp.where((t < 0.0) | (t > tf_right), right_home_pos, interp_fun(t, right_time_array, right_xpos_array))

    def right_grip_command_cb(time):
        t = time - t0
        return jnp.where((t > right_time_array[2]) & (t < right_time_array[5]), 0.0, 255.0)

    left_pos_source = library.SourceBlock(left_pos_command_cb, name="left_pos_command")
    left_grip_source = library.SourceBlock(left_grip_command_cb, name="left_grip_command")

    right_pos_source = library.SourceBlock(right_pos_command_cb, name="right_pos_command")
    right_grip_source = library.SourceBlock(right_grip_command_cb, name="right_grip_command")
    
    system = BuildSystem(
    mjmodel,
    right_grip_source,
    right_pos_source,
    left_grip_source,
    left_pos_source,
    left_arm_actuator_ids,
    right_arm_actuator_ids,
    dt
    )
    
    results = RunSystem(
    system,
    mjmodel,
    left_pos_source,
    right_pos_source,
    tf
    )
    
    file_name = "position_error_tracking.png"
    
    SavePlot(
    results,
    file_name
    )
    
    RenderVideo(
    model,
    mjmodel,
    results,
    data,
    left_target_id,
    right_target_id,
    tf
)

if __name__ == '__main__': 
    main()
