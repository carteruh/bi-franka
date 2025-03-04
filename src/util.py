import numpy as np
import mediapy
import mujoco
import matplotlib.pyplot as plt
from collimator import library


def RenderVideo(
    model,
    mjmodel,
    results,
    data,
    left_target_id,
    right_target_id,
    tf
):
    # Render the simulation
    fps = 60.0
    t = np.arange(0, tf, 1/fps)

    qpos = np.zeros((len(t), mjmodel.nq))
    for i in range(mjmodel.nq):
        qpos[:, i] = np.interp(t, results.time, results.outputs["qpos"][:, i])

    left_block_xpos = np.zeros((len(t), 3))
    right_block_xpos = np.zeros((len(t), 3))

    for i in range(3):
        left_block_xpos[:, i] = np.interp(t, results.time, results.outputs["left_target_xpos"][:, i])
        right_block_xpos[:, i] = np.interp(t, results.time, results.outputs["right_target_xpos"][:, i])

    left_block_xmat = np.zeros((len(t), 9))
    right_block_xmat = np.zeros((len(t), 9))

    for i in range(9):
        left_block_xmat[:, i] = np.interp(t, results.time, results.outputs["left_target_xmat"][:, i])
        right_block_xmat[:, i] = np.interp(t, results.time, results.outputs["right_target_xmat"][:, i])

    frames = np.zeros((len(t), *mjmodel._video_default.shape), dtype=np.uint8)
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        data.site_xpos[left_target_id] = left_block_xpos[i]
        data.site_xmat[left_target_id] = left_block_xmat[i]

        data.site_xpos[right_target_id] = right_block_xpos[i]
        data.site_xmat[right_target_id] = right_block_xmat[i]
        mujoco.mj_kinematics(model, data)
        frames[i] = mjmodel.render(data)

    mediapy.write_video("output.mp4", frames, fps=fps)

def SavePlot(
    results,
    file_name
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 2), sharex=True)

    for i in range(3):
        ax.plot(results.time, results.outputs["left_pos_cmd"][:, i], c='k')
        ax.plot(results.time, results.outputs["left_hand_xpos"][:, i], '--', c='tab:red')
        ax.plot(results.time, results.outputs["right_pos_cmd"][:, i], c='b')
        ax.plot(results.time, results.outputs["right_hand_xpos"][:, i], '--', c='tab:blue')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.legend(["Commanded", "Actual"])
    plt.savefig(file_name)
    
def ConfigureMujoco(
    model,
    dt,
    xml_file,
    left_hand_id,
    right_hand_id,
    left_target_id,
    right_target_id
):
    left_jac_buffer = np.zeros((6, model.nv))
    right_jac_buffer = np.zeros((6, model.nv))

    def left_jac_script(model, data):
        mujoco.mj_jacBody(model, data, left_jac_buffer[:3], left_jac_buffer[3:], left_hand_id)
        return left_jac_buffer

    def right_jac_script(model, data):
        mujoco.mj_jacBody(model, data, right_jac_buffer[:3], right_jac_buffer[3:], right_hand_id)
        return right_jac_buffer



    custom_outputs = {
        "left_hand_xpos": lambda model, data: data.body(left_hand_id).xpos,
        "left_hand_xquat": lambda model, data: data.body(left_hand_id).xquat,
        "left_target_xpos": lambda model, data: data.site(left_target_id).xpos,
        "left_target_xmat": lambda model, data: data.site(left_target_id).xmat,
        "left_jac": left_jac_script,

        "right_hand_xpos": lambda model, data: data.body(right_hand_id).xpos,
        "right_hand_xquat": lambda model, data: data.body(right_hand_id).xquat,
        "right_target_xpos": lambda model, data: data.site(right_target_id).xpos,
        "right_target_xmat": lambda model, data: data.site(right_target_id).xmat,
        "right_jac": right_jac_script
    }



    mjmodel = library.mujoco.MuJoCo(
        file_name=xml_file,
        dt=dt,
        enable_video_output=True,
        custom_output_scripts=custom_outputs,
        key_frame_0="home"
    )
    data = mjmodel._data
    
    return data, mjmodel