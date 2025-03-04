from scipy.spatial import transform


import collimator
from collimator import library
from collimator.backend import io_callback

from .inverse_kinematics import InverseKinematics



def BuildSystem(
    mjmodel,
    right_grip_source,
    right_pos_source,
    left_grip_source,
    left_pos_source,
    left_arm_actuator_ids,
    right_arm_actuator_ids,
    dt
    
):
    builder = collimator.DiagramBuilder()
    builder.add(mjmodel, right_grip_source, right_pos_source, left_grip_source, left_pos_source )

    # Constant orientation for the end-effector
    rot = transform.Rotation.from_euler("xyz", [90, 0, 180], degrees=True)
    const_q0 = builder.add(library.Constant(rot.as_quat(), name="q0"))

    # Create separate IK blocks for left and right arms
    ik_left  = builder.add(InverseKinematics(nv=mjmodel.nv, name="ik_left", dt=dt))
    ik_left.name = "ik_left"
    ik_right  = builder.add(InverseKinematics(nv=mjmodel.nv, name="ik_right", dt=dt))
    ik_right.name = "ik_right"


    # Left arm IK connections (each block expects 5 inputs)
    builder.connect(left_pos_source.output_ports[0], ik_left.input_ports[0])   # left target pos
    builder.connect(const_q0.output_ports[0], ik_left.input_ports[1])          # left target quat
    builder.connect(mjmodel.get_output_port("left_hand_xpos"), ik_left.input_ports[2])   # current pos
    builder.connect(mjmodel.get_output_port("left_hand_xquat"), ik_left.input_ports[3])  # current quat
    builder.connect(mjmodel.get_output_port("left_jac"), ik_left.input_ports[4])          # Jacobian
    builder.connect(mjmodel.get_output_port("right_hand_xpos"), ik_left.input_ports[5])


    # Right arm IK connections (each block expects 5 inputs)
    builder.connect(right_pos_source.output_ports[0], ik_right.input_ports[0])   # right target pos
    builder.connect(const_q0.output_ports[0], ik_right.input_ports[1])           # right target quat
    builder.connect(mjmodel.get_output_port("right_hand_xpos"), ik_right.input_ports[2])   # current pos
    builder.connect(mjmodel.get_output_port("right_hand_xquat"), ik_right.input_ports[3])  # current quat
    builder.connect(mjmodel.get_output_port("right_jac"), ik_right.input_ports[4])          # Jacobian
    builder.connect(mjmodel.get_output_port("left_hand_xpos"), ik_right.input_ports[5])


    # (Remove any duplicate block that connects 10 ports to a single "ik" block)

    # Extract only the controlled joints from each IK block's output.
    left_dq_arm = builder.add(library.FeedthroughBlock(lambda dq: dq[left_arm_actuator_ids],
                                                        name="left_dq_arm"))
    builder.connect(ik_left.output_ports[0], left_dq_arm.input_ports[0])

    right_dq_arm = builder.add(library.FeedthroughBlock(lambda dq: dq[right_arm_actuator_ids],
                                                        name="right_dq_arm"))
    builder.connect(ik_right.output_ports[0], right_dq_arm.input_ports[0])

    # Demultiplex the joint commands for each arm.
    left_demux_dq = builder.add(library.Demultiplexer(len(left_arm_actuator_ids), name="left_demux_dq"))
    builder.connect(left_dq_arm.output_ports[0], left_demux_dq.input_ports[0])

    right_demux_dq = builder.add(library.Demultiplexer(len(right_arm_actuator_ids), name="right_demux_dq"))
    builder.connect(right_dq_arm.output_ports[0], right_demux_dq.input_ports[0])

    # Create separate PID controllers for each joint.
    # Create separate PID controllers for each arm
    left_pid_controllers = []
    right_pid_controllers = []

    for i in range(len(left_arm_actuator_ids)):
        pid_left = builder.add(library.PIDDiscrete(kp=.1, ki=1.3, kd=0.008, dt=dt, name=f"left_pid_{i}"))
        builder.connect(left_demux_dq.output_ports[i], pid_left.input_ports[0])
        left_pid_controllers.append(pid_left)

    for i in range(len(right_arm_actuator_ids)):
        pid_right = builder.add(library.PIDDiscrete(kp=.1, ki=1.3, kd=0.008, dt=dt, name=f"right_pid_{i}"))
        builder.connect(right_demux_dq.output_ports[i], pid_right.input_ports[0])
        right_pid_controllers.append(pid_right)

    # Combine the PID outputs for each arm
    left_pid_outputs = builder.add(library.Multiplexer(len(left_arm_actuator_ids), name="left_pid_outputs"))
    right_pid_outputs = builder.add(library.Multiplexer(len(right_arm_actuator_ids), name="right_pid_outputs"))

    for i, pid_left in enumerate(left_pid_controllers):
        builder.connect(pid_left.output_ports[0], left_pid_outputs.input_ports[i])

    for i, pid_right in enumerate(right_pid_controllers):
        builder.connect(pid_right.output_ports[0], right_pid_outputs.input_ports[i])

    # Add the base control signal to the PID output for each arm
    left_adder_uq = builder.add(library.Adder(2, name="left_adder_uq"))
    left_const_uq0 = builder.add(library.Constant(mjmodel.ctrl_0[left_arm_actuator_ids], name="left_uq0"))
    builder.connect(left_const_uq0.output_ports[0], left_adder_uq.input_ports[0])
    builder.connect(left_pid_outputs.output_ports[0], left_adder_uq.input_ports[1])

    right_adder_uq = builder.add(library.Adder(2, name="right_adder_uq"))
    right_const_uq0 = builder.add(library.Constant(mjmodel.ctrl_0[right_arm_actuator_ids], name="right_uq0"))
    builder.connect(right_const_uq0.output_ports[0], right_adder_uq.input_ports[0])
    builder.connect(right_pid_outputs.output_ports[0], right_adder_uq.input_ports[1])

    # Append the gripper command to the joint commands for each arm
    left_mux_u = builder.add(library.Multiplexer(2, name="left_mux_u"))
    builder.connect(left_adder_uq.output_ports[0], left_mux_u.input_ports[0])
    builder.connect(left_grip_source.output_ports[0], left_mux_u.input_ports[1])

    right_mux_u = builder.add(library.Multiplexer(2, name="right_mux_u"))
    builder.connect(right_adder_uq.output_ports[0], right_mux_u.input_ports[0])
    builder.connect(right_grip_source.output_ports[0], right_mux_u.input_ports[1])

    # Combine the control signals for both arms
    final_u = builder.add(library.Multiplexer(2, name="final_u"))
    builder.connect(left_mux_u.output_ports[0], final_u.input_ports[0])
    builder.connect(right_mux_u.output_ports[0], final_u.input_ports[1])

    # Connect final control to the model's input
    builder.connect(final_u.output_ports[0], mjmodel.input_ports[0])

    # Build the system
    system = builder.build()
    system.pprint()
    return system

def RunSystem(
    system,
    mjmodel,
    left_pos_source,
    right_pos_source,
    tf
):
    context = system.create_context()

    recorded_signals = {
        "qpos": mjmodel.output_ports[0],
        "left_pos_cmd": left_pos_source.output_ports[0],
        "left_hand_xpos": mjmodel.get_output_port("left_hand_xpos"),
        "left_target_xpos": mjmodel.get_output_port("left_target_xpos"),
        "left_target_xmat": mjmodel.get_output_port("left_target_xmat"),

        "right_pos_cmd": right_pos_source.output_ports[0],
        "right_hand_xpos": mjmodel.get_output_port("right_hand_xpos"),
        "right_target_xpos": mjmodel.get_output_port("right_target_xpos"),
        "right_target_xmat": mjmodel.get_output_port("right_target_xmat"),
    }

    results = collimator.simulate(
        system,
        context,
        (0.0, tf),
        recorded_signals=recorded_signals,
    ) 
    
    return results