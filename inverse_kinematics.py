# inverse_kinematics.py

import numpy as np
import mujoco
from collimator.framework import LeafSystem
from collimator.backend import io_callback

class IKController(LeafSystem):
    """
    An IK block that:
      - Has input ports: target_pos, target_quat, pos, quat, jac
      - Outputs joint velocities (nv,).
    """

    def __init__(self, nv, dt=0.01, name="InverseKinematics"):
        super().__init__(name=name)

        # We'll store these for dimensioning
        self.nv = nv
        self.dt = dt

        # Preallocate arrays for computing inverse kinematics
        self.error = np.zeros(6)
        self.hand_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)

        # Input ports
        self.declare_input_port("target_pos")    # shape (3,)
        self.declare_input_port("target_quat")   # shape (4,)
        self.declare_input_port("pos")           # shape (3,)
        self.declare_input_port("quat")          # shape (4,)
        self.declare_input_port("jac")           # shape (6, nv)

        # Output port -> joint velocities
        def _ik_output_cb(time, state, *inputs, **parameters):
            # We pass a default array of zeros to io_callback for shape (nv,)
            return io_callback(self._diffik, np.zeros(self.nv), *inputs)

        # Name this port something, e.g. "joint_vel"
        # The 'period=dt' means we'll recalc at each time step.
        self.declare_output_port(_ik_output_cb, name="joint_vel", period=dt)

    def _diffik(self, target_pos, target_quat, pos, quat, jac):
        """
        target_pos: (3,)
        target_quat: (4,)
        pos: (3,)
        quat: (4,)
        jac: (6, nv)
        Returns: (nv,) joint velocities
        """
        # 1) Position error
        self.error[:3] = target_pos - pos

        # 2) Orientation error
        mujoco.mju_negQuat(self.hand_quat_conj, quat)
        mujoco.mju_mulQuat(self.error_quat, target_quat, self.hand_quat_conj)
        mujoco.mju_quat2Vel(self.error[3:], self.error_quat, 1.0)

        # 3) Solve J dq = error via pseudoinverse
        dq = np.linalg.pinv(jac) @ self.error
        return dq
