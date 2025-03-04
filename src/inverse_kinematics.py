# inverse_kinematics.py

import numpy as np
import mujoco
from collimator.framework import LeafSystem
from collimator.backend import io_callback
import collimator

class InverseKinematics(collimator.LeafSystem):
    def __init__(self, nv, name="InverseKinematics",  dt=.001):
        super().__init__(name=name)
        # Preallocate arrays for computing inverse kinematics
        self.error = np.zeros(6)
        self.hand_quat = np.zeros(4)
        self.hand_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
        self.dt = dt

        # Create input ports
        self.declare_input_port("target_pos")
        self.declare_input_port("target_quat")
        self.declare_input_port("pos")
        self.declare_input_port("quat")
        self.declare_input_port("jac")
        
        self.declare_input_port("other_pos")  # e.g. a 3D vector for the other arm's EEF


        # Output port: joint velocities
        def _output_cb(time, state, *inputs, **parameters):
            return io_callback(self._diffik, np.zeros(nv), *inputs)

        self.declare_output_port(_output_cb, period=self.dt)

    def _diffik(self,
                target_pos, target_quat,
                pos, quat,
                jac,
                other_pos):
        # 1) Standard IK error
        self.error[:3] = target_pos - pos
        mujoco.mju_negQuat(self.hand_quat_conj, quat)
        mujoco.mju_mulQuat(self.error_quat, target_quat, self.hand_quat_conj)
        mujoco.mju_quat2Vel(self.error[3:], self.error_quat, 1.0)

        # 2) Repulsive push if within d_min
        d_min = 0.10  # 10 cm min distance
        alpha = 0.5   # repulsion gain

        diff = pos - other_pos  # vector from other EEF to this EEF
        dist = np.linalg.norm(diff)
        if dist < d_min:
            # push away from the other EEF
            self.error[:3] += alpha * (d_min - dist) * (diff / dist)

        return  np.linalg.pinv(jac) @ self.error