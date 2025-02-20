# diagram_setup.py

import numpy as np
from collimator.framework import DiagramBuilder
from collimator.framework.error import CollimatorError
from collimator.framework import LeafSystem

import mujoco
from inverse_kinematics import IKController


class SimpleMuJoCoBlock(LeafSystem):
    """
    Minimal block:
      - Loads scene.xml (or your model path)
      - Interprets its 'control' input as joint velocities
      - Publishes example output ports: pos, quat, jac, etc.
    For demonstration, we only create some placeholders.
    """
    def __init__(self, xml_path="/home/carter/bimanual-franka-main/resources/bi-franka/scene.xml", dt=0.01):
        super().__init__(name="MuJoCoBlock")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.dt = dt

        self.nv = self.model.nv
        # Let's assume we have a site called 'right_ee_site'
        self.ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_ee_site")

        # 1) Input port for joint velocities (size = nv)
        self.declare_input_port("control")  # We'll interpret this as velocity

        # 2) We'll define some output ports for demonstration:
        #    a) "pos" (3,) -> end-effector position
        #    b) "quat" (4,) -> end-effector orientation
        #    c) "jac" (6,nv) -> 6D Jacobian
        # Typically you'd want "target_pos" and "target_quat" from somewhere else,
        # but let's just define them here for a minimal example. 
        self._default_pos = np.zeros(3)
        self._default_quat = np.array([1, 0, 0, 0])
        self._default_jac = np.zeros((6, self.nv))

        self.declare_output_port(self._pos_cb, name="pos", default_value=self._default_pos)
        self.declare_output_port(self._quat_cb, name="quat", default_value=self._default_quat)
        self.declare_output_port(self._jac_cb, name="jac", default_value=self._default_jac)

        # Also declare "target_pos" and "target_quat" outputs if you want the IK to consume them:
        self._desired_pos = np.array([1.0, 0.3, 0.8])       # Example
        self._desired_quat = np.array([1, 0, 0, 0])         # Identity
        self.declare_output_port(self._tpos_cb, name="target_pos", default_value=self._desired_pos)
        self.declare_output_port(self._tquat_cb, name="target_quat", default_value=self._desired_quat)

    def _pos_cb(self, t, state, *inputs, **params) -> np.ndarray:
        self._step_physics()
        return self.data.site_xpos[self.ee_site].copy()

    def _quat_cb(self, t, state, *inputs, **params) -> np.ndarray:
        self._step_physics()
        return self.data.site_xquat[self.ee_site].copy()

    def _jac_cb(self, t, state, *inputs, **params) -> np.ndarray:
        self._step_physics()
        jacp = np.zeros((3, self.nv))
        jacr = np.zeros((3, self.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site)
        return np.vstack([jacp, jacr])

    def _tpos_cb(self, t, state, *inputs, **params) -> np.ndarray:
        return self._desired_pos

    def _tquat_cb(self, t, state, *inputs, **params) -> np.ndarray:
        return self._desired_quat

    def _step_physics(self):
        """
        1) Read velocity from 'control' input, apply to self.data.qvel
        2) Step once. 
        """
        vel = self.get_input_port("control").latest_input()
        if vel is None:
            vel = np.zeros(self.nv)
        self.data.qvel[:] = vel
        # Step the model
        mujoco.mj_step(self.model, self.data)


def build_diagram():
    builder = DiagramBuilder()

    # Add the MuJoCo block
    mj_block = SimpleMuJoCoBlock(dt=0.01)
    builder.add(mj_block)

    # Add the IK block (we pass mj_block.nv for the dimension)
    ik_block = IKController(nv=mj_block.nv, dt=0.01)
    builder.add(ik_block)

    # Now connect them using the 2-argument method: connect(src_output, dest_input)

    # 1) MuJoCo -> IK
    #   target_pos -> IK target_pos
    builder.connect(
        mj_block.get_output_port("target_pos"),
        ik_block.get_input_port("target_pos")
    )
    #   target_quat -> IK target_quat
    builder.connect(
        mj_block.get_output_port("target_quat"),
        ik_block.get_input_port("target_quat")
    )
    #   pos -> IK pos
    builder.connect(
        mj_block.get_output_port("pos"),
        ik_block.get_input_port("pos")
    )
    #   quat -> IK quat
    builder.connect(
        mj_block.get_output_port("quat"),
        ik_block.get_input_port("quat")
    )
    #   jac -> IK jac
    builder.connect(
        mj_block.get_output_port("jac"),
        ik_block.get_input_port("jac")
    )

    # 2) IK -> MuJoCo
    #   IK "joint_vel" -> MuJoCo "control"
    # builder.connect(
    #     ik_block.get_output_port("joint_vel"),
    #     mj_block.get_input_port("control")
    # )

    # Build the diagram
    diagram = builder.build()
    return diagram
