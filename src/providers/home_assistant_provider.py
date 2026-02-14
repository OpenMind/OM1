import logging
import os
from typing import Any, Dict, List, Optional

import aiohttp

from providers.singleton import singleton

logger = logging.getLogger(__name__)


@singleton
class HomeAssistantProvider:
    """
    Singleton provider for Home Assistant REST API communication.

    Shared by both action connectors and input plugins to interact
    with a Home Assistant instance.

    Parameters
    ----------
    base_url : str
        Base URL of the Home Assistant instance.
    token : str
        Long-lived access token for authentication.
    token_env : str
        Environment variable name containing the access token.
    timeout_seconds : int
        HTTP request timeout in seconds.
    verify_ssl : bool
        Whether to verify SSL certificates.
    """

    def __init__(
        self,
        base_url: str = "http://homeassistant.local:8123",
        token: str = "",
        token_env: str = "HOME_ASSISTANT_TOKEN",
        timeout_seconds: int = 10,
        verify_ssl: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.token_env = token_env
        self.timeout_seconds = timeout_seconds
        self.verify_ssl = verify_ssl

    def _get_token(self) -> str:
        """
        Retrieve the authentication token.

        Checks the environment variable first, then falls back to the
        directly configured token.

        Returns
        -------
        str
            The authentication token.

        Raises
        ------
        ValueError
            If no token is available from either source.
        """
        env_token = os.environ.get(self.token_env, "")
        if env_token:
            return env_token
        if self.token:
            return self.token
        raise ValueError(
            f"No Home Assistant token found. Set the '{self.token_env}' "
            f"environment variable or provide a token in the config."
        )

    def _headers(self) -> Dict[str, str]:
        """
        Build HTTP headers for Home Assistant API requests.

        Returns
        -------
        Dict[str, str]
            Headers dictionary with Authorization and Content-Type.
        """
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
        }

    async def get_state(self, entity_id: str) -> Dict[str, Any]:
        """
        Get the current state of a single entity.

        Parameters
        ----------
        entity_id : str
            The entity ID to query (e.g. "light.living_room").

        Returns
        -------
        Dict[str, Any]
            The entity state object from Home Assistant.

        Raises
        ------
        RuntimeError
            If the HTTP request fails.
        """
        url = f"{self.base_url}/api/states/{entity_id}"
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        ssl = None if self.verify_ssl else False

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=self._headers(), ssl=ssl) as response:
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(
                        f"Failed to get state for {entity_id}: "
                        f"{response.status} {text}"
                    )
                return await response.json()

    async def get_states(
        self, entity_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the current states of multiple entities.

        If entity_ids is provided, only those entities are returned.
        Otherwise, all entities are returned.

        Parameters
        ----------
        entity_ids : Optional[List[str]]
            List of entity IDs to filter. If None, returns all.

        Returns
        -------
        List[Dict[str, Any]]
            List of entity state objects.

        Raises
        ------
        RuntimeError
            If the HTTP request fails.
        """
        url = f"{self.base_url}/api/states"
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        ssl = None if self.verify_ssl else False

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=self._headers(), ssl=ssl) as response:
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(
                        f"Failed to get states: {response.status} {text}"
                    )
                all_states = await response.json()

        if entity_ids is not None:
            entity_set = set(entity_ids)
            return [s for s in all_states if s.get("entity_id") in entity_set]

        return all_states

    async def call_service(
        self,
        domain: str,
        service: str,
        entity_id: str,
        **kwargs: Any,
    ) -> None:
        """
        Call a Home Assistant service.

        Parameters
        ----------
        domain : str
            The service domain (e.g. "light", "climate").
        service : str
            The service name (e.g. "turn_on", "turn_off").
        entity_id : str
            The target entity ID.
        **kwargs : Any
            Additional service data fields.

        Raises
        ------
        RuntimeError
            If the HTTP request fails.
        """
        url = f"{self.base_url}/api/services/{domain}/{service}"
        payload: Dict[str, Any] = {"entity_id": entity_id}
        payload.update(kwargs)

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        ssl = None if self.verify_ssl else False

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url, headers=self._headers(), json=payload, ssl=ssl
            ) as response:
                if response.status not in (200, 201):
                    text = await response.text()
                    raise RuntimeError(
                        f"Failed to call {domain}.{service} on {entity_id}: "
                        f"{response.status} {text}"
                    )
