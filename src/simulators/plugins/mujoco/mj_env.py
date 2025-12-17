from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._mujoco_typing import MjData, MjModel, mj_name2id, mj_resetData, mj_step, mjtObj


@dataclass
class MJEnv:
    model_path: str

    def __post_init__(self):
        self.model = MjModel.from_xml_path(self.model_path)
        self.data = MjData(self.model)

    def reset(self):
        mj_resetData(self.model, self.data)
        return self._obs()

    def set_target_qpos(self, target: dict):
        """
        target contoh: {"j1": 15, "j2": -30}  # derajat
        """
        for name, deg in target.items():
            j_id = mj_name2id(self.model, mjtObj.mjOBJ_JOINT, name)
            a_id = self._actuator_for_joint(j_id)
            if a_id >= 0:
                self.data.ctrl[a_id] = np.deg2rad(float(deg))

    def step(self, n: int = 1):
        n = int(n)
        for _ in range(n):
            mj_step(self.model, self.data)
        return self._obs()

    def _actuator_for_joint(self, joint_id: int) -> int:
        for i in range(self.model.nu):
            if self.model.actuator_trnid[i, 0] == joint_id:
                return i
        return -1

    def _obs(self):
        qpos = np.array(self.data.qpos).tolist()
        qvel = np.array(self.data.qvel).tolist()
        return {"qpos": qpos, "qvel": qvel, "time": float(self.data.time)}
