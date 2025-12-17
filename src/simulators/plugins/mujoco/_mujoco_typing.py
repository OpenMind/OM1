from typing import Any, MutableSequence


class MjModel:
    nu: int
    actuator_trnid: Any

    @classmethod
    def from_xml_path(cls, path: str) -> "MjModel": ...


class MjData:
    ctrl: MutableSequence[float]
    qpos: MutableSequence[float]
    qvel: MutableSequence[float]
    time: float

    def __init__(self, model: MjModel) -> None: ...


def mj_resetData(model: MjModel, data: MjData) -> None: ...
def mj_step(model: MjModel, data: MjData) -> None: ...
def mj_name2id(model: MjModel, objtype: Any, name: str) -> int: ...


class mjtObj:
    mjOBJ_JOINT: int
