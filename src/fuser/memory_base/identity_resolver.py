import logging


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
