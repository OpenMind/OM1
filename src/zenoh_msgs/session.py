import logging
from typing import Any, Callable, Optional, Union

import zenoh

from zenoh_msgs.cloud_session.topic_schemas import CLOUD_TOPICS
from zenoh_msgs.cloud_session.zenoh_adapter import (
    CloudSimZenohSession,
    _Publisher,
    _Sample,
    _Subscriber,
)

ZenohSessionType = Union[zenoh.Session, CloudSimZenohSession, "HybridZenohSession", "_ZenohSessionContextManager"]
ZenohSampleType = Union[zenoh.Sample, "_Sample"]


def create_zenoh_config(network_discovery: bool = True) -> zenoh.Config:
    """
    Create a Zenoh configuration for a client connecting to a Zenoh router.

    Parameters
    ----------
    network_discovery : bool, optional
        Whether to enable network discovery (default is True).

    Returns
    -------
    zenoh.Config
        The Zenoh configuration object.
    """
    config = zenoh.Config()
    if not network_discovery:
        config.insert_json5("mode", '"client"')
        config.insert_json5("connect/endpoints", '["tcp/127.0.0.1:7447"]')

    return config


class HybridZenohSession:
    """
    A session wrapper that routes subscriptions and publications to either a cloud
    simulation session or a standard Zenoh session based on the topic.

    Topics in CLOUD_TOPICS are routed to the cloud session, while all other topics
    are routed to the standard Zenoh session.
    """

    _api_key: Optional[str] = None
    _use_sim: bool = False

    def __init__(self) -> None:
        """
        Initialize the hybrid session with lazy-initialized underlying sessions.
        """
        self._cloud_session: Optional[CloudSimZenohSession] = None
        self._zenoh_session: Optional[zenoh.Session] = None
        self._api_key = HybridZenohSession._api_key
        self._use_sim = HybridZenohSession._use_sim

    @classmethod
    def set_api_key(cls, api_key: str) -> None:
        """
        Set the API key for all HybridZenohSession instances.

        Parameters
        ----------
        api_key : str
            The default API key to use for cloud session authentication.
        """
        cls._api_key = api_key

    @classmethod
    def set_use_sim(cls, use_sim: bool) -> None:
        """
        Set the simulation mode for all HybridZenohSession instances.

        Parameters
        ----------
        use_sim : bool
            Whether the system is running in simulation mode.
        """
        cls._use_sim = use_sim

    @property
    def use_sim(self) -> bool:
        """
        Get the simulation mode.

        Returns
        -------
        bool
            Whether the system is running in simulation mode.
        """
        return self._use_sim

    def _get_cloud_session(self) -> CloudSimZenohSession:
        """
        Get or create the cloud session.

        Returns
        -------
        CloudSimZenohSession
            The cloud simulation session for topics in CLOUD_TOPICS.
        """
        if self._cloud_session is None:
            self._cloud_session = _open_cloud_session(token=self._api_key)
            logging.info("Initialized cloud simulation session")
        return self._cloud_session

    def _get_zenoh_session(self) -> zenoh.Session:
        """
        Get or create the standard Zenoh session.

        Returns
        -------
        zenoh.Session
            The standard Zenoh session for non-cloud topics.
        """
        if self._zenoh_session is None:
            local_config = create_zenoh_config(network_discovery=False)
            try:
                self._zenoh_session = zenoh.open(local_config)
                logging.info("Zenoh client opened without network discovery")
            except Exception:
                logging.info("Falling back to network discovery...")
                config = create_zenoh_config()
                try:
                    self._zenoh_session = zenoh.open(config)
                    logging.info("Zenoh client opened with network discovery")
                except Exception as e:
                    logging.error(f"Error opening Zenoh client: {e}")
                    raise Exception("Failed to open Zenoh session") from e
        return self._zenoh_session

    def declare_subscriber(
        self,
        topic: str,
        handler: Optional[Callable[[Any], None]],
    ) -> Optional[Union[_Subscriber, zenoh.Subscriber]]:
        """
        Subscribe to a topic, routing to the appropriate underlying session.

        Parameters
        ----------
        topic : str
            The topic to subscribe to.
        handler : Callable[[Any], None]
            A callback function to handle incoming messages.

        Returns
        -------
        Optional[Union[_Subscriber, zenoh.Subscriber]]
            A subscriber object from the appropriate session.
        """
        if topic in CLOUD_TOPICS:
            logging.debug(f"Routing topic '{topic}' to cloud session")
            return self._get_cloud_session().declare_subscriber(topic, handler)
        else:
            logging.debug(f"Routing topic '{topic}' to standard Zenoh session")
            return self._get_zenoh_session().declare_subscriber(topic, handler)

    def declare_publisher(self, topic: str) -> Optional[Union[_Publisher, zenoh.Publisher]]:
        """
        Create a publisher for a topic, routing to the appropriate underlying session.

        Parameters
        ----------
        topic : str
            The topic to publish to.

        Returns
        -------
        Optional[Union[_Publisher, zenoh.Publisher]]
            A publisher object from the appropriate session.
        """
        if topic in CLOUD_TOPICS:
            logging.debug(f"Routing topic '{topic}' to cloud session")
            return self._get_cloud_session().declare_publisher(topic)
        else:
            logging.debug(f"Routing topic '{topic}' to standard Zenoh session")
            return self._get_zenoh_session().declare_publisher(topic)

    def get(self, topic: str, *args: Any, **kwargs: Any) -> Any:
        """
        Perform a get operation on a topic, routing to the appropriate underlying session.

        Note: Query/response patterns (get) are not supported on cloud simulation topics.
        If a cloud topic is requested, this will use the standard Zenoh session instead.

        Parameters
        ----------
        topic : str
            The topic to query.
        *args : Any
            Additional positional arguments to pass to the underlying session.
        **kwargs : Any
            Additional keyword arguments to pass to the underlying session.

        Returns
        -------
        Any
            The result from the underlying session's get operation.
        """
        if topic in CLOUD_TOPICS:
            logging.warning(
                f"Query/response (get) not supported for cloud topic '{topic}', "
                "falling back to standard Zenoh session"
            )
        logging.debug(f"Routing get for topic '{topic}' to standard Zenoh session")
        return self._get_zenoh_session().get(topic, *args, **kwargs)

    def put(self, topic: str, payload: Any, *args: Any, **kwargs: Any) -> None:
        """
        Publish to a topic, routing to the appropriate underlying session.

        Parameters
        ----------
        topic : str
            The topic to publish to.
        payload : Any
            The payload to publish.
        *args : Any
            Additional positional arguments to pass to the underlying session.
        **kwargs : Any
            Additional keyword arguments to pass to the underlying session.
        """
        if topic in CLOUD_TOPICS:
            logging.debug(f"Routing put for topic '{topic}' to cloud session")
            self._get_cloud_session().put(topic, payload, *args, **kwargs)
        else:
            logging.debug(f"Routing put for topic '{topic}' to standard Zenoh session")
            self._get_zenoh_session().put(topic, payload, *args, **kwargs)

    def close(self) -> None:
        """
        Close both underlying sessions if they were initialized.
        """
        if self._cloud_session is not None:
            self._cloud_session.close()
            logging.info("Closed cloud simulation session")
        if self._zenoh_session is not None:
            self._zenoh_session.close()
            logging.info("Closed standard Zenoh session")


def open_zenoh_session() -> "_ZenohSessionContextManager":
    """
    Open a Zenoh session.

    Returns
    -------
    _ZenohSessionContextManager
        A context manager that yields a Zenoh session.
    """
    return _ZenohSessionContextManager()


class _ZenohSessionContextManager:
    """
    Context manager wrapper for Zenoh sessions.

    This class enables using open_zenoh_session() with the 'with' statement,
    ensuring proper cleanup of resources. It also maintains backward compatibility
    by allowing direct attribute access without using a context manager.
    """

    def __init__(self) -> None:
        """
        Initialize the context manager without creating the session yet.
        """
        self._session: Optional[ZenohSessionType] = None

    def __enter__(self) -> ZenohSessionType:
        """
        Open and return the Zenoh session.

        Returns
        -------
        ZenohSessionType
            The opened Zenoh session.
        """
        self._session = _create_zenoh_session()

        return self._session

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        """
        Close the Zenoh session.

        Parameters
        ----------
        _exc_type : type, optional
            The type of exception that occurred, if any.
        _exc_val : Exception, optional
            The exception instance that occurred, if any.
        _exc_tb : traceback, optional
            The traceback object, if any.
        """
        if self._session is not None:
            self._session.close()
            logging.info("Closed Zenoh session")

    def __getattr__(self, name: str) -> Any:
        """
        Forward attribute access to the underlying session.

        Parameters
        ----------
        name : str
            The name of the attribute to access.

        Returns
        -------
        Any
            The value of the requested attribute from the underlying session.
        """
        if self._session is None:
            self._session = _create_zenoh_session()
        return getattr(self._session, name)


def _create_zenoh_session() -> ZenohSessionType:
    """
    Internal function to create a Zenoh session.

    Returns
    -------
    ZenohSessionType
        A session object that can be used to interact with Zenoh topics.
    """
    if HybridZenohSession._use_sim:
        return HybridZenohSession()

    local_config = create_zenoh_config(network_discovery=False)
    try:
        session = zenoh.open(local_config)
        logging.info("Zenoh client opened without network discovery")
        return session
    except Exception:
        logging.info("Falling back to network discovery...")

    config = create_zenoh_config()
    try:
        session = zenoh.open(config)
        logging.info("Zenoh client opened with network discovery")
        return session
    except Exception as e:
        logging.error(f"Error opening Zenoh client: {e}")
        raise Exception("Failed to open Zenoh session") from e


def _open_cloud_session(
    url: str = "wss://api.openmind.com/api/core/simulation/zenoh", token: Optional[str] = None
) -> CloudSimZenohSession:
    """
    Open a CloudSimZenohSession using the URL and token from environment variables.
    """
    return CloudSimZenohSession(url, token=token)


def load_api_key(api_key: Optional[str]):
    """
    Load the API key for cloud session authentication.

    This function sets the API key for all HybridZenohSession instances, allowing
    them to authenticate with the cloud broker when routing topics in CLOUD_TOPICS.

    Parameters
    ----------
    api_key : str
        The API key to use for cloud session authentication.
    """
    if not api_key:
        return

    HybridZenohSession.set_api_key(api_key)


def load_use_sim(use_sim: bool):
    """
    Load the simulation mode setting.

    This function sets the simulation mode for all HybridZenohSession instances,
    allowing them to route topics in CLOUD_TOPICS to the cloud session when use_sim is True.

    Parameters
    ----------
    use_sim : bool
        Whether the system is running in simulation mode.
    """
    HybridZenohSession.set_use_sim(use_sim)


def load_session_config(api_key: Optional[str], use_sim: bool):
    """
    Load session configuration settings.

    This function sets both the API key and simulation mode for all HybridZenohSession instances.

    Parameters
    ----------
    api_key : Optional[str]
        The API key to use for cloud session authentication.
    use_sim : bool
        Whether the system is running in simulation mode.
    """
    load_api_key(api_key)
    load_use_sim(use_sim)


if __name__ == "__main__":
    session = open_zenoh_session()
    if session:
        logging.info("Session opened successfully")
        session.close()
    else:
        logging.error("Failed to open Zenoh session")
