import json
import logging
import shutil
from pathlib import Path
from typing import Optional

DEFAULT_GALLERY_ROOT = "/data/gallery"
DEFAULT_EMBEDS_ROOT = "/data/embeds"
EMBEDS_SUBDIR = "arc-trt-l2-512"


def resolve_current_user() -> str:
    """Resolve the current user_id from IOProvider's cached face presence data.

    Parses the 'Closest: xxx.' field from the FacePresence input text stored
    in IOProvider, avoiding extra HTTP requests.

    Returns 'unknown' if no face data is available or if the closest face is unknown.
    """
    try:
        from providers.io_provider import IOProvider

        io = IOProvider()
        face_input = io.get_input("FacePresence")
        if not face_input or not face_input.input:
            return "unknown"

        text = face_input.input
        marker = "Closest: "
        idx = text.find(marker)
        if idx < 0:
            return "unknown"

        name = text[idx + len(marker) :].rstrip(".").strip().lower()
        if not name or name == "unknown":
            return "unknown"
        return name
    except Exception as e:
        logging.debug(f"Identity resolver: could not resolve user: {e}")
        return "unknown"


def get_face_photo(user_id: str, gallery_root: str = DEFAULT_GALLERY_ROOT) -> Optional[Path]:
    """Find the first image file in the gallery directory for a user.

    Parameters
    ----------
    user_id : str
        Normalized user identity.
    gallery_root : str
        Root directory of the face gallery.

    Returns
    -------
    Path or None
        Path to the first image found, or None.
    """
    if user_id == "unknown":
        return None

    gallery_dir = Path(gallery_root) / user_id
    if not gallery_dir.is_dir():
        return None

    for ext in ("*.jpg", "*.jpeg", "*.png"):
        matches = sorted(gallery_dir.glob(ext))
        if matches:
            return matches[0]
    return None


def get_face_embedding(user_id: str, embeds_root: str = DEFAULT_EMBEDS_ROOT) -> Optional[list[float]]:
    """Read the face embedding vector for a user from the embeds index.

    Parameters
    ----------
    user_id : str
        Normalized user identity.
    embeds_root : str
        Root directory of the face embeddings cache.

    Returns
    -------
    list[float] or None
        512-dim embedding vector, or None if not found.
    """
    if user_id == "unknown":
        return None

    index_path = Path(embeds_root) / EMBEDS_SUBDIR / "index.json"
    if not index_path.exists():
        return None

    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        items = data.get("items", {})
        entry = items.get(user_id)
        if entry and "embedding" in entry:
            return list(entry["embedding"])
    except Exception as e:
        logging.warning(f"Identity resolver: failed to read embedding for {user_id}: {e}")

    return None


def copy_face_photo(user_id: str, dest_dir: Path, gallery_root: str = DEFAULT_GALLERY_ROOT) -> Optional[str]:
    """Copy the user's gallery photo into the memory user directory.

    Parameters
    ----------
    user_id : str
        Normalized user identity.
    dest_dir : Path
        Destination directory (memory/users/{user_id}/).
    gallery_root : str
        Root directory of the face gallery.

    Returns
    -------
    str or None
        Relative path from memory root (e.g. 'users/alice/face.jpg'), or None.
    """
    dest_file = dest_dir / "face.jpg"
    if dest_file.exists():
        return f"users/{user_id}/face.jpg"

    source = get_face_photo(user_id, gallery_root)
    if source is None:
        return None

    try:
        shutil.copy2(source, dest_file)
        logging.info(f"Identity resolver: copied face photo for {user_id}")
        return f"users/{user_id}/face.jpg"
    except Exception as e:
        logging.warning(f"Identity resolver: failed to copy face photo for {user_id}: {e}")
        return None
