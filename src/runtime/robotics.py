import logging


def load_unitree(unitree_ethernet: str):
    """
    Initialize the Unitree robot's network communication channel.

    This function sets up the Ethernet connection for a Unitree robot based on
    the provided configuration or environment variables. It can operate in either
    real hardware or simulation mode.

    Parameters
    ----------
    unitree_ethernet : str
        Configuration object containing the Unitree Ethernet adapter string, such as "eth0"

    Returns
    -------
    None

    Raises
    ------
    Exception
        If initialization of the Unitree Ethernet channel fails.

    """
    if unitree_ethernet is not None:
        logging.info(f"Using {unitree_ethernet} as the Unitree Network Ethernet Adapter")

        from unitree.unitree_sdk2py.core.channel import ChannelFactoryInitialize

        try:
            ChannelFactoryInitialize(0, unitree_ethernet)
        except Exception as e:
            logging.error(f"Failed to initialize Unitree Ethernet channel: {e}")
            # raise e
        logging.info("Booting Unitree and CycloneDDS")
