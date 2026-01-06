#!/usr/bin/env python3
import json5, hashlib, time, logging
from pathlib import Path
from typing import Callable, Dict
from threading import Thread, Lock

logger = logging.getLogger(__name__)

class ConfigWatcher:
    def __init__(self, config_dir: str = "config", poll_interval: int = 2, hot_reload_enabled: bool = True):
        self.config_dir = Path(config_dir)
        self.poll_interval = poll_interval
        self.enabled = hot_reload_enabled
        self._file_hashes = {}
        self._observers = {}
        self._lock = Lock()
        self._watcher_thread = None
        self._running = False
        self._update_all_hashes()
    
    def _get_file_hash(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except:
            return ""
    
    def _update_all_hashes(self):
        if self.config_dir.exists():
            for f in self.config_dir.glob("*.json5"):
                self._file_hashes[str(f)] = self._get_file_hash(f)
    
    def register_observer(self, config_name: str, callback: Callable):
        with self._lock:
            self._observers[config_name] = callback
    
    def _load_config(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json5.load(f)
        except:
            return {}
    
    def _watch_loop(self):
        while self._running:
            try:
                if self.config_dir.exists():
                    for f in self.config_dir.glob("*.json5"):
                        current = self._get_file_hash(f)
                        stored = self._file_hashes.get(str(f), "")
                        if current != stored and current:
                            self._handle_config_change(f)
                            self._file_hashes[str(f)] = current
                time.sleep(self.poll_interval)
            except:
                time.sleep(self.poll_interval)
    
    def _handle_config_change(self, filepath):
        config_name = filepath.stem
        new_config = self._load_config(filepath)
        if new_config and config_name in self._observers:
            self._observers[config_name](new_config)
    
    def start(self):
        if not self.enabled or self._running:
            return
        self._running = True
        self._watcher_thread = Thread(target=self._watch_loop, daemon=True)
        self._watcher_thread.start()
    
    def stop(self):
        self._running = False
        if self._watcher_thread:
            self._watcher_thread.join(timeout=5)
