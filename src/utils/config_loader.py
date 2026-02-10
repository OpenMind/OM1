
from src.utils.api_key_manager import get_api_key, mask_api_key
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load config with secure API key handling"""
    
    # Load JSON5 config
    config = json5.load(open(config_path))
    
    # Replace API keys from secure storage
    if 'brain' in config and 'llm' in config['brain']:
        llm_config = config['brain']['llm']
        provider = llm_config.get('provider', '').lower()
        
        # Check if API key in config is a placeholder
        config_key = llm_config.get('api_key', '')
        
        if config_key in ['openmind_free', 'your_api_key_here', '']:
            # Load from secure storage
            api_key = get_api_key(provider)
            
            if api_key:
                llm_config['api_key'] = api_key
                logger.info(f"Loaded API key for {provider}: {mask_api_key(api_key)}")
            else:
                logger.error(f"No API key found for {provider}")
                raise ValueError(f"API key required for {provider}")
        else:
            # Warn about plain text key
            logger.warning(
                f"⚠️  API key in config file (plain text). "
                f"Consider using environment variables or keychain."
            )
    
    return config
