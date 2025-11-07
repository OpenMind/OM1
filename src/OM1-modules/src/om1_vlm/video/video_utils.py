from typing import List, Tuple

import cv2


def enumerate_video_devices(index: int = 0) -> List[Tuple[int, str]]:
    """
    Enumerate available video capture devices on the system.

    Iteratively attempts to open video capture devices starting from the given
    index until no more devices can be opened.

    Parameters
    ----------
    index : int, optional
        Starting index for device enumeration, by default 0

    Returns
    -------
    List[Tuple[int, str]]
        List of tuples containing:
        - Device index (int)
        - Backend name (str)
    """
    devices = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        devices.append((index, cap.getBackendName()))
        cap.release()
        index += 1
    return devices
