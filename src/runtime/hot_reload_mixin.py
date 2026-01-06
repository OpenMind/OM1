#!/usr/bin/env python3
import logging
from src.runtime.config_watcher import ConfigWatcher

logger = logging.getLogger(__name__)

class HotReloadMixin:
    HOT_RELOAD_FIELDS = {'system_prompt_base', 'system_prompt', 'temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'hashtags', 'tweet_interval', 'engagement_enabled', 'max_retweets_per_day', 'daily_budget_usd'}
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_watcher = None
        self._hot_reload_enabled = kwargs.get('hot_reload_enabled', True)
    
    def setup_hot_reload(self, config_name: str):
        if not self._hot_reload_enabled:
            return
        self.config_watcher = ConfigWatcher(config_dir="config", poll_interval=2, hot_reload_enabled=True)
        self.config_watcher.register_observer(config_name, self._on_config_reload)
        self.config_watcher.start()
    
    def _on_config_reload(self, new_config):
        for field in self.HOT_RELOAD_FIELDS:
            if field in new_config:
                old_value = getattr(self, field, None)
                new_value = new_config[field]
                if old_value != new_value:
                    setattr(self, field, new_value)
        if hasattr(self, '_post_hot_reload'):
            self._post_hot_reload(new_config)
    
    def shutdown_hot_reload(self):
        if self.config_watcher:
            self.config_watcher.stop()
