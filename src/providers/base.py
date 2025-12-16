from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ProviderBase(ABC):
    name: str

    @abstractmethod
    def init(self, config: Optional[Dict[str, Any]] = None) -> None: ...

    @abstractmethod
    def connect(self, **kwargs) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_balance(self, asset: str) -> float: ...

    @abstractmethod
    def sign_message(self, message: str) -> str: ...
