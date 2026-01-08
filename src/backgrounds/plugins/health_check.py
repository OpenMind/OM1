"""
Health Check Background
监控系统健康状态并定期记录关键指标
"""

import logging
import psutil
import threading
import time
from datetime import datetime

from backgrounds.base import Background, BackgroundConfig


class HealthCheck(Background):
    """
    System health monitor that tracks and logs key metrics.
    
    Monitors:
    - CPU usage percentage
    - Memory usage (RSS and percentage)
    - Disk I/O statistics
    - Network connection count
    - Process uptime
    
    Useful for:
    - Production monitoring
    - Performance debugging
    - Resource optimization
    """

    def __init__(self, config: BackgroundConfig = BackgroundConfig()):
        super().__init__(config)

        # Configuration options
        self.check_interval = getattr(config, "check_interval", 60)  # seconds
        self.enable_logging = getattr(config, "enable_logging", True)
        self.alert_threshold_cpu = getattr(config, "alert_threshold_cpu", 90.0)
        self.alert_threshold_memory = getattr(config, "alert_threshold_memory", 90.0)

        if not self.enable_logging:
            return

        # Track start time
        self.start_time = time.time()
        self.process = psutil.Process()
        self.last_disk_io = psutil.disk_io_counters()
        self.last_net_io = psutil.net_io_counters()
        self.last_check_time = time.time()

        # Start health check thread
        self._running = True
        self._health_thread = threading.Thread(
            target=self._health_check_loop, daemon=True
        )
        self._health_thread.start()

        logging.info(f"✅ HealthCheck: Started (interval={self.check_interval}s)")

    def _health_check_loop(self):
        """Run health checks at configured interval."""
        while self._running:
            try:
                self._check_health()
            except Exception as e:
                logging.error(f"HealthCheck error: {e}")
            
            time.sleep(self.check_interval)

    def _check_health(self):
        """Perform health check and log metrics."""
        current_time = time.time()
        uptime_seconds = current_time - self.start_time

        # CPU metrics
        cpu_percent = self.process.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()

        # Memory metrics
        mem_info = self.process.memory_info()
        mem_percent = self.process.memory_percent()
        system_mem = psutil.virtual_memory()

        # Disk I/O (since last check)
        disk_io = psutil.disk_io_counters()
        disk_read_mb = 0
        disk_write_mb = 0
        if self.last_disk_io and disk_io:
            time_delta = current_time - self.last_check_time
            if time_delta > 0:
                disk_read_mb = (disk_io.read_bytes - self.last_disk_io.read_bytes) / (1024 * 1024)
                disk_write_mb = (disk_io.write_bytes - self.last_disk_io.write_bytes) / (1024 * 1024)

        # Network I/O (since last check)
        net_io = psutil.net_io_counters()
        net_sent_mb = 0
        net_recv_mb = 0
        if self.last_net_io and net_io:
            time_delta = current_time - self.last_check_time
            if time_delta > 0:
                net_sent_mb = (net_io.bytes_sent - self.last_net_io.bytes_sent) / (1024 * 1024)
                net_recv_mb = (net_io.bytes_recv - self.last_net_io.bytes_recv) / (1024 * 1024)

        # Update last check values
        self.last_disk_io = disk_io
        self.last_net_io = net_io
        self.last_check_time = current_time

        # Thread count
        num_threads = self.process.num_threads()

        # Format uptime
        uptime_hours = int(uptime_seconds // 3600)
        uptime_minutes = int((uptime_seconds % 3600) // 60)
        uptime_str = f"{uptime_hours}h {uptime_minutes}m"

        # Log health status
        health_info = (
            f"📊 HealthCheck [{uptime_str}] | "
            f"CPU: {cpu_percent:.1f}% ({cpu_count} cores) | "
            f"Memory: {mem_percent:.1f}% ({mem_info.rss / (1024**2):.0f}MB / {system_mem.total / (1024**3):.1f}GB) | "
            f"Threads: {num_threads} | "
            f"Disk R/W: {disk_read_mb:.1f}/{disk_write_mb:.1f} MB/s | "
            f"Net TX/RX: {net_sent_mb:.1f}/{net_recv_mb:.1f} MB/s"
        )
        logging.info(health_info)

        # Alert if thresholds exceeded
        if cpu_percent > self.alert_threshold_cpu:
            logging.warning(f"⚠️  High CPU usage: {cpu_percent:.1f}%")
        
        if mem_percent > self.alert_threshold_memory:
            logging.warning(f"⚠️  High memory usage: {mem_percent:.1f}%")

    def get_health_metrics(self) -> dict:
        """
        Get current health metrics as a dictionary.
        Useful for external monitoring tools.
        
        Returns
        -------
        dict
            Current system health metrics
        """
        current_time = time.time()
        uptime_seconds = current_time - self.start_time
        
        return {
            "uptime_seconds": uptime_seconds,
            "cpu_percent": self.process.cpu_percent(interval=0.1),
            "memory_mb": self.process.memory_info().rss / (1024**2),
            "memory_percent": self.process.memory_percent(),
            "threads": self.process.num_threads(),
            "timestamp": datetime.now().isoformat()
        }
