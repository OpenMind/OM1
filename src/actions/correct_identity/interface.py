# src/actions/correct_identity/interface.py
from dataclasses import dataclass

from actions.base import Interface


@dataclass
class CorrectIdentityInput:
    """
    Input to rename or merge a recently-enrolled identity.

    Used when the LLM enrolled the WRONG LABEL on a real person (typo,
    speech misrecognition, etc.) — same physical person, wrong name.

    Distinct from:
      - selfie(force=True)   → different physical person who looks similar
      - forget_last()         → wrong physical person was captured entirely

    Parameters
    ----------
    from_id : str
        The current (incorrect) identity name in the gallery.
    to_id : str
        The desired identity name. If to_id already exists, samples are
        merged into it. If not, from_id's folder is renamed to to_id.

        Both must follow API naming rules: lowercase ASCII alphanumeric,
        dash, underscore. The trailing `_<digits>` suffix is reserved.

    Notes
    -----
    Only works within ~60s of the source enrollment (the face API
    enforces a TTL on `last_enrollment`). Outside that window, this
    action returns `result=stale_enrollment` and the LLM should
    apologize rather than retry.
    """

    from_id: str
    to_id: str


@dataclass
class CorrectIdentity(Interface[CorrectIdentityInput, CorrectIdentityInput]):
    """
    Rename a recently-enrolled identity to fix a label error.

    Trigger conditions (all must hold):
      - SelfieStatus showed `result=success id=<from_id>` within ~60s
      - User's wording indicates label correction, NOT a different person
        ("Actually it's John, you misheard me" / "Sorry, J-O-H-N")
      - User does NOT mention a third party

    If those conditions don't hold, prefer selfie(force=True) for
    look-alikes or forget_last() for wrong-person captures.
    """

    input: CorrectIdentityInput
    output: CorrectIdentityInput
