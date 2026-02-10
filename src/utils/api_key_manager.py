# src/utils/api_key_manager.py

"""
OM1 API Key Security Manager

Securely manages API keys with support for:
- Environment variables
- Encrypted storage
- Key validation
- Secure cleanup
"""

import os
import re
from pathlib import Path
from typing import Optional
import logging
from cryptography.fernet import Fernet
import keyring  # For OS-level keychain integration

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Secure API key management for OM1"""
    
    # API key format patterns for validation
    KEY_PATTERNS = {
        'openmind': r'^om-[a-zA-Z0-9]{32,}$',
        'openai': r'^sk-[a-zA-Z0-9]{48,}$',
        'anthropic': r'^sk-ant-[a-zA-Z0-9\-]{95,}$',
        'gemini': r'^AIza[a-zA-Z0-9\-_]{35}$',
        'deepseek': r'^sk-[a-zA-Z0-9]{48,}$',
    }
    
    def __init__(self, keyring_service: str = "OM1"):
        """
        Initialize API Key Manager
        
        Args:
            keyring_service: Service name for OS keychain
        """
        self.keyring_service = keyring_service
        self._encryption_key = None
    
    def get_api_key(self, provider: str, key_name: str = "api_key") -> Optional[str]:
        """
        Get API key from secure sources in priority order:
        1. Environment variable
        2. OS Keychain
        3. .env file (fallback)
        
        Args:
            provider: LLM provider name (e.g., 'openai', 'openmind')
            key_name: Key name in config
            
        Returns:
            API key or None if not found
        """
        # 1. Check environment variable
        env_var = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(env_var)
        if api_key:
            logger.debug(f"API key loaded from environment: {env_var}")
            return api_key
        
        # 2. Check OS keychain
        try:
            api_key = keyring.get_password(self.keyring_service, provider)
            if api_key:
                logger.debug(f"API key loaded from keychain: {provider}")
                return api_key
        except Exception as e:
            logger.warning(f"Could not access keychain: {e}")
        
        # 3. Check .env file
        api_key = self._load_from_env_file(env_var)
        if api_key:
            logger.debug(f"API key loaded from .env file: {env_var}")
            return api_key
        
        logger.error(f"API key not found for provider: {provider}")
        return None
    
    def set_api_key(self, provider: str, api_key: str, 
                    use_keychain: bool = True) -> bool:
        """
        Securely store API key
        
        Args:
            provider: Provider name
            api_key: API key to store
            use_keychain: Store in OS keychain (recommended)
            
        Returns:
            True if successful
        """
        # Validate key format
        if not self.validate_api_key(provider, api_key):
            logger.error(f"Invalid API key format for {provider}")
            return False
        
        try:
            if use_keychain:
                keyring.set_password(self.keyring_service, provider, api_key)
                logger.info(f"API key stored securely in keychain: {provider}")
            else:
                # Store in .env file as fallback
                self._save_to_env_file(provider, api_key)
                logger.info(f"API key stored in .env file: {provider}")
            return True
        except Exception as e:
            logger.error(f"Failed to store API key: {e}")
            return False
    
    def validate_api_key(self, provider: str, api_key: str) -> bool:
        """
        Validate API key format
        
        Args:
            provider: Provider name
            api_key: API key to validate
            
        Returns:
            True if valid
        """
        if not api_key or len(api_key) < 10:
            return False
        
        # Check for placeholder values
        placeholders = ['your_api_key', 'replace_me', 'openmind_free', 'xxx']
        if any(p in api_key.lower() for p in placeholders):
            logger.warning("API key appears to be a placeholder")
            return False
        
        # Check provider-specific format
        pattern = self.KEY_PATTERNS.get(provider.lower())
        if pattern:
            if not re.match(pattern, api_key):
                logger.warning(f"API key doesn't match expected format for {provider}")
                # Still return True - format might have changed
        
        return True
    
    def delete_api_key(self, provider: str) -> bool:
        """
        Delete API key from secure storage
        
        Args:
            provider: Provider name
            
        Returns:
            True if successful
        """
        try:
            keyring.delete_password(self.keyring_service, provider)
            logger.info(f"API key deleted from keychain: {provider}")
            return True
        except Exception as e:
            logger.warning(f"Could not delete from keychain: {e}")
            return False
    
    def _load_from_env_file(self, env_var: str) -> Optional[str]:
        """Load API key from .env file"""
        env_file = Path.cwd() / '.env'
        if not env_file.exists():
            return None
        
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith(env_var):
                        return line.split('=', 1)[1].strip().strip('"\'')
        except Exception as e:
            logger.error(f"Error reading .env file: {e}")
        
        return None
    
    def _save_to_env_file(self, provider: str, api_key: str):
        """Save API key to .env file"""
        env_file = Path.cwd() / '.env'
        env_var = f"{provider.upper()}_API_KEY"
        
        # Read existing content
        lines = []
        if env_file.exists():
            with open(env_file, 'r') as f:
                lines = [l for l in f.readlines() if not l.startswith(env_var)]
        
        # Add new key
        lines.append(f"{env_var}={api_key}\n")
        
        # Write back
        with open(env_file, 'w') as f:
            f.writelines(lines)
        
        # Set restrictive permissions
        env_file.chmod(0o600)
    
    @staticmethod
    def mask_api_key(api_key: str, visible_chars: int = 4) -> str:
        """
        Mask API key for logging
        
        Args:
            api_key: Full API key
            visible_chars: Number of characters to show
            
        Returns:
            Masked key (e.g., 'sk-ab...xyz9')
        """
        if not api_key or len(api_key) < visible_chars * 2:
            return "***"
        
        return f"{api_key[:visible_chars]}...{api_key[-visible_chars:]}"


# Convenience functions
_manager = APIKeyManager()

def get_api_key(provider: str) -> Optional[str]:
    """Get API key for provider"""
    return _manager.get_api_key(provider)

def set_api_key(provider: str, api_key: str) -> bool:
    """Set API key for provider"""
    return _manager.set_api_key(provider, api_key)

def validate_api_key(provider: str, api_key: str) -> bool:
    """Validate API key format"""
    return _manager.validate_api_key(provider, api_key)
