"""
Auto Restart Background
监控系统崩溃或错误并自动重启
"""

import logging
import threading
import time
from datetime import datetime

from backgrounds.base import Background, BackgroundConfig


class AutoRestart(Background):
    """
    Automatic restart handler for system crashes and errors.
    
    Features:
    - Monitors system health via heartbeat
    - Automatic restart on prolonged unresponsiveness
    - Configurable crash detection thresholds
    - Restart history tracking
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        self.check_interval = getattr(config, "check_interval", 30)
        self.crash_threshold = getattr(config, "crash_threshold", 300)  # 5 minutes
        self.max_restarts = getattr(config, "max_restarts", 3)
        self.restart_window = getattr(config, "restart_window", 3600)  # 1 hour

        self.last_heartbeat = time.time()
        self.restart_count = 0
        self.restart_history = []
        self._running = True

        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True
        )
        self._monitor_thread.start()

        logging.info("✅ AutoRestart: Started crash monitoring")

    def update_heartbeat(self):
        """Update the heartbeat timestamp. Call this periodically to indicate system is alive."""
        self.last_heartbeat = time.time()

    def _monitor_loop(self):
        """Monitor system health and restart if needed."""
        while self._running:
            try:
                self._check_system_health()
            except Exception as e:
                logging.error(f"AutoRestart monitoring error: {e}")
            time.sleep(self.check_interval)

    def _check_system_health(self):
        """Check if system is responsive and restart if crashed."""
        current_time = time.time()
        time_since_heartbeat = current_time - self.last_heartbeat

        # Check if system appears crashed
        if time_since_heartbeat > self.crash_threshold:
            logging.warning(
                f"⚠️  System unresponsive for {time_since_heartbeat:.0f}s "
                f"(threshold: {self.crash_threshold}s)"
            )

            # Check restart rate limiting
            self._cleanup_old_restarts(current_time)
            if len(self.restart_history) >= self.max_restarts:
                logging.error(
                    f"❌ Too many restarts ({len(self.restart_history)} in "
                    f"{self.restart_window}s). Auto-restart disabled."
                )
                return

            # Attempt restart
            self._attempt_restart(current_time)

    def _cleanup_old_restarts(self, current_time):
        """Remove restart history entries outside the time window."""
        cutoff = current_time - self.restart_window
        self.restart_history = [
            t for t in self.restart_history if t > cutoff
        ]

    def _attempt_restart(self, current_time):
        """Attempt to restart the system."""
        self.restart_history.append(current_time)
        self.restart_count += 1

        logging.info(
            f"🔄 Attempting auto-restart #{self.restart_count} "
            f"(last heartbeat {current_time - self.last_heartbeat:.0f}s ago)"
        )

        # Trigger restart via parent system
        try:
            if hasattr(self, 'trigger_restart'):
                self.trigger_restart()
            else:
                logging.warning("AutoRestart: No restart handler available")
        except Exception as e:
            logging.error(f"AutoRestart failed: {e}")

    def get_restart_stats(self) -> dict:
        """Get restart statistics."""
        return {
            "total_restarts": self.restart_count,
            "recent_restarts": len(self.restart_history),
            "last_heartbeat": self.last_heartbeat,
            "time_since_heartbeat": time.time() - self.last_heartbeat,
            "history": [
                datetime.fromtimestamp(t).isoformat()
                for t in self.restart_history[-10:]  # Last 10
            ]
        }
