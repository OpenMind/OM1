# src/actions/selfie/interface.py
from dataclasses import dataclass

from actions.base import Interface


@dataclass
class SelfieInput:
    """
    Input to enroll a person via the multi-frame /selfie endpoint.

    Parameters
    ----------
    action : str
        The identity name (e.g., "wendy"). Determines what the API does:
          - New name → create gallery/<id> with 1-4 captured samples
          - Same-name family match (e.g. "wendy" vs existing "wendy_1")
            with cosine ≥ merge_the → merge samples into existing folder
          - Cross-name match (different name, cosine ≥ cross_name_the)
            → reject as `face_belongs_to`, unless `force=True`

        Naming rules: lowercase ASCII alphanumeric, dash, underscore.
        Trailing `_<digits>` is reserved for the dedup system — don't use it.

    timeout_sec : int
        Max seconds to wait for at least one face to appear before giving up.
        Default 5. Bump higher (e.g. 10) for hesitant users. Note: the /selfie
        API itself takes ~1.5s once started (multi-frame collection window),
        so total worst-case latency is roughly `timeout_sec + 1.5s`.

    force : bool
        If True, bypass the cross-name reject. Use when the user disputes
        a matched identity, e.g.:
            User: "I'm not Wendy, I'm John"
            LLM:  selfie(action="john", force=True)
        Default False — normal enrollments should leave this off so the
        dedup safety net works.
    """

    action: str
    timeout_sec: int = 5
    force: bool = False


@dataclass
class Selfie(Interface[SelfieInput, SelfieInput]):
    """
    Enroll a person to the face gallery via the multi-frame /selfie endpoint.

    The endpoint collects 1-4 quality-gated frames over a ~1.5s window,
    selects the best target by engagement score (face area × frontality),
    and either creates a new identity or merges into an existing one.

    See SelfieInput for parameters. Outcomes are surfaced to the LLM via the
    SelfieStatus input, with a brief TTS confirmation sent to the user.
    """

    input: SelfieInput
    output: SelfieInput
