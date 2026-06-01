# src/actions/forget_last/interface.py
from dataclasses import dataclass
from typing import Optional

from actions.base import Interface


@dataclass
class ForgetLastInput:
    """
    Input to undo the most recent /selfie enrollment.

    Used when the WRONG PERSON was captured during enrollment — e.g., a
    different person walked in front of the camera, so the saved samples
    belong to someone other than who the LLM was talking to.

    Distinct from:
      - correct_identity()  → right person, wrong label (typo)
      - selfie(force=True)  → different person who looks similar

    Parameters
    ----------
    id : str, optional
        Safety check. If provided, must equal the API's last_enrollment.id;
        the call fails with `result=id_mismatch` otherwise. Omit to undo
        whatever the most recent enrollment was regardless of id.

    Notes
    -----
    Only works within ~60s of the enrollment.
    """

    id: Optional[str] = None


@dataclass
class ForgetLast(Interface[ForgetLastInput, ForgetLastInput]):
    """
    Undo the most recent enrollment by deleting its samples.

    Trigger conditions:
      - SelfieStatus showed a recent `result=success` (≤60s)
      - User indicates the WRONG PERSON was captured
        ("You saw the person behind me" / "That wasn't me" /
         "Someone walked in front of the camera")

    NOT for:
      - Typos / mishearings → use correct_identity()
      - Look likes (different person, similar face) → use selfie(force=True)
    """

    input: ForgetLastInput
    output: ForgetLastInput
