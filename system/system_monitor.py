"""System resource monitoring with support for local and remote machines."""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger('roo_bench.system_monitor')


@dataclass
class SystemStats:
    """Статистика использования ресурсов системы."""
    
    # CPU метрики
    cpu_percent: float = 0.0
    cpu_percent_samples: list[float] = field(default_factory=list)
    
    # RAM метрики
    ram_used: int = 0
    ram_total: int = 0
    ram_percent: float = 0.0
    ram_samples: list[int] = field(default_factory=list)
    
    # VRAM метрики
    vram_used: int | None = None
    vram_total: int | None = None
    vram_percent: float | None = None
    vram_samples: list[int] = field(default_factory=list)
    
    # Временные метки
    timestamp: float = 0.0
    
    def add_sample(self, cpu_percent: float, ram_used: int, vram_used: int | None = None) -> None:
        """Добавить новую выборку метрик."""
        self.cpu_percent_samples.append(cpu_percent)
        self.ram_samples.append(ram_used)
        if vram_used is not None:
            self.vram_samples.append(vram_used)
        
        # Ограничение размера выборок для экономии памяти
        max_samples = 1000
        if len(self.cpu_percent_samples) > max_samples:
            self.cpu_percent_samples = self.cpu_percent_samples[-max_samples:]
        if len(self.ram_samples) > max_samples:
            self.ram_samples = self.ram_samples[-max_samples:]
        if len(self.vram_samples) > max_samples:
            self.vram_samples = self.vram_samples[-max_samples:]
    
    def get_aggregated_stats(self) -> dict[str, Any]:
        """Получить агрегированную статистику."""
        stats = {
            'cpu': {
                'current': self.cpu_percent,
                'avg': sum(self.cpu_percent_samples) / len(self.cpu_percent_samples) if self.cpu_percent_samples else 0,
                'min': min(self.cpu_percent_samples) if self.cpu_percent_samples else 0,
                'max': max(self.cpu_percent_samples) if self.cpu_percent_samples else 0,
                'samples_count': len(self.cpu_percent_samples)
            },
            'ram': {
                'used_current': self.ram_used,
                'total': self.ram_total,
                'percent_current': self.ram_percent,
                'avg_percent': sum(s / self.ram_total * 100 for s in self.ram_samples) / len(self.ram_samples) if self.ram_samples else 0,
                'min_percent': min(s / self.ram_total * 100 for s in self.ram_samples) if self.ram_samples else 0,
                'max_percent': max(s / self.ram_total * 100 for s in self.ram_samples) if self.ram_samples else 0,
                'samples_count': len(self.ram_samples)
            }
        }
        
        if self.vram_used is not None and self.vram_total is not None:
            vram_stats: dict[str, Any] = {
                'used_current': self.vram_used,
                'total': self.vram_total,
                'percent_current': self.vram_percent if self.vram_percent is not None else 0.0,
                'avg_percent': sum(s / self.vram_total * 100 for s in self.vram_samples) / len(self.vram_samples) if self.vram_samples else 0,
                'min_percent': min(s / self.vram_total * 100 for s in self.vram_samples) if self.vram_samples else 0,
                'max_percent': max(s / self.vram_total * 100 for s in self.vram_samples) if self.vram_samples else 0,
                'samples_count': len(self.vram_samples)
            }
            stats['vram'] = vram_stats
        
        return stats


class BaseMonitor(ABC):
    """Базовый класс для мониторинга ресурсов."""
    
    def __init__(self, collection_interval: float = 0.5) -> None:
        """Инициализация монитора.
        
        Args:
            collection_interval: Интервал сбора метрик в секундах
        """
        self.collection_interval = collection_interval
        self.is_monitoring = False
        self.monitor_thread: threading.Thread | None = None
        self.stats_history: list[SystemStats] = []
        self._lock = threading.Lock()
    
    @abstractmethod
    def collect_stats(self) -> SystemStats | None:
        """Собрать текущие метрики системы.
        
        Returns:
            SystemStats с текущими метриками или None если сбор не удался
        """
        pass
    
    def start_monitoring(self, duration: float | None = None) -> None:
        """Начать мониторинг ресурсов.
        
        Args:
            duration: Длительность мониторинга в секундах (None для бесконечности)
        """
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.stats_history = []
        
        def monitor_loop() -> None:
            start_time = time.time()
            last_collection = 0.0
            
            while self.is_monitoring:
                current_time = time.time()
                
                # Проверка длительности
                if duration and (current_time - start_time) >= duration:
                    break
                
                # Сбор метрик с заданным интервалом
                if current_time - last_collection >= self.collection_interval:
                    stats = self.collect_stats()
                    if stats:
                        with self._lock:
                            self.stats_history.append(stats)
                    last_collection = current_time
                
                time.sleep(0.1)  # Небольшая пауза для снижения нагрузки
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        if self.monitor_thread is not None:
            self.monitor_thread.start()
        logger.info(f"Started monitoring with interval {self.collection_interval}s")
    
    def stop_monitoring(self) -> None:
        """Остановить мониторинг."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        logger.info("Stopped monitoring")
    
    def get_latest_stats(self) -> SystemStats | None:
        """Получить последние собранные метрики."""
        with self._lock:
            return self.stats_history[-1] if self.stats_history else None
    
    def get_aggregated_stats(self) -> dict[str, Any] | None:
        """Получить агрегированную статистику по всем собранным метрикам."""
        if not self.stats_history:
            return None
        
        # Агрегация по всем метрикам
        all_cpu_samples = []
        all_ram_samples = []
        all_vram_samples = []
        
        for stats in self.stats_history:
            all_cpu_samples.extend(stats.cpu_percent_samples)
            all_ram_samples.extend(stats.ram_samples)
            if stats.vram_samples:
                all_vram_samples.extend(stats.vram_samples)
        
        if not all_cpu_samples:
            return None
        
        aggregated = {
            'cpu': {
                'avg': sum(all_cpu_samples) / len(all_cpu_samples),
                'min': min(all_cpu_samples),
                'max': max(all_cpu_samples),
                'samples_count': len(all_cpu_samples)
            },
            'ram': {
                'avg_percent': sum(s / self.stats_history[0].ram_total * 100 for s in all_ram_samples) / len(all_ram_samples) if all_ram_samples else 0,
                'min_percent': min(s / self.stats_history[0].ram_total * 100 for s in all_ram_samples) if all_ram_samples else 0,
                'max_percent': max(s / self.stats_history[0].ram_total * 100 for s in all_ram_samples) if all_ram_samples else 0,
                'samples_count': len(all_ram_samples)
            }
        }
        
        if all_vram_samples and self.stats_history[0].vram_total:
            aggregated['vram'] = {
                'avg_percent': sum(s / self.stats_history[0].vram_total * 100 for s in all_vram_samples) / len(all_vram_samples),
                'min_percent': min(s / self.stats_history[0].vram_total * 100 for s in all_vram_samples),
                'max_percent': max(s / self.stats_history[0].vram_total * 100 for s in all_vram_samples),
                'samples_count': len(all_vram_samples)
            }
        
        return aggregated
    
    def get_stats_history(self) -> list[SystemStats]:
        """Получить историю собранных метрик."""
        with self._lock:
            return self.stats_history.copy()


class LocalSystemMonitor(BaseMonitor):
    """Мониторинг ресурсов локальной машины."""
    
    psutil: Any = None
    gpu_monitor: dict[str, Any] | None = None
    
    def __init__(self, collection_interval: float = 0.5) -> None:
        """Инициализация локального монитора."""
        super().__init__(collection_interval)
        self._try_import_psutil()
        self._try_import_gpu_monitor()
    
    def _try_import_psutil(self) -> None:
        """Попытка импорта psutil для мониторинга CPU/RAM."""
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            logger.warning("psutil not available, CPU/RAM monitoring disabled")
            self.psutil = None
    
    def _try_import_gpu_monitor(self) -> None:
        """Попытка импорта gpu_monitor для мониторинга VRAM."""
        try:
            from .gpu_monitor import check_gpu_available, get_vram_usage
            self.gpu_monitor = {
                'get_vram_usage': get_vram_usage,
                'check_gpu_available': check_gpu_available
            }
        except ImportError:
            logger.warning("gpu_monitor not available, VRAM monitoring disabled")
            self.gpu_monitor = None
    
    def collect_stats(self) -> SystemStats | None:
        """Собрать текущие метрики локальной системы."""
        try:
            # CPU метрики
            cpu_percent = 0.0
            if self.psutil:
                cpu_percent = self.psutil.cpu_percent(interval=0.1)
            
            # RAM метрики
            ram_used = 0
            ram_total = 0
            ram_percent = 0.0
            if self.psutil:
                memory = self.psutil.virtual_memory()
                ram_used = memory.used
                ram_total = memory.total
                ram_percent = memory.percent
            
            # VRAM метрики
            vram_used = None
            vram_total = None
            vram_percent = None
            if self.gpu_monitor and self.gpu_monitor['check_gpu_available']():
                vram_bytes = self.gpu_monitor['get_vram_usage']()
                if vram_bytes is not None:
                    # Получаем общий объем VRAM через nvidia-smi
                    try:
                        import subprocess
                        result = subprocess.run(
                            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'],
                            capture_output=True, text=True, timeout=5
                        )
                        if result.returncode == 0:
                            vram_total = int(result.stdout.strip()) * 1024 * 1024
                            vram_used = vram_bytes
                            vram_percent = (vram_used / vram_total) * 100 if vram_total > 0 else 0
                    except Exception:
                        pass
            
            timestamp = time.time()
            
            stats = SystemStats(
                cpu_percent=cpu_percent,
                ram_used=ram_used,
                ram_total=ram_total,
                ram_percent=ram_percent,
                vram_used=vram_used,
                vram_total=vram_total,
                vram_percent=vram_percent,
                timestamp=timestamp
            )
            
            # Добавляем выборки для агрегации
            stats.add_sample(cpu_percent, ram_used, vram_used)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error collecting local system stats: {e}")
            return None


class RemoteSystemMonitor(BaseMonitor):
    """Мониторинг ресурсов удаленной машины через SSH."""
    
    def __init__(self, ssh_client: Any, collection_interval: float = 0.5) -> None:
        """Инициализация удаленного монитора.
        
        Args:
            ssh_client: SSHClient instance для удаленных операций
            collection_interval: Интервал сбора метрик в секундах
        """
        super().__init__(collection_interval)
        self.ssh_client = ssh_client
    
    def collect_stats(self) -> SystemStats | None:
        """Собрать текущие метрики удаленной системы."""
        if not self.ssh_client or not self.ssh_client.is_configured:
            logger.warning("SSH client not configured for remote monitoring")
            return None
        
        try:
            # CPU метрики
            cpu_percent = self._get_cpu_usage_remote()
            
            # RAM метрики
            ram_stats = self._get_ram_usage_remote()
            ram_used = ram_stats.get('used', 0) if ram_stats else 0
            ram_total = ram_stats.get('total', 0) if ram_stats else 0
            ram_percent = ram_stats.get('percent', 0.0) if ram_stats else 0.0
            
            # VRAM метрики
            vram_used = self.ssh_client.get_vram_usage()
            vram_total = None
            vram_percent = None
            
            if vram_used is not None:
                # Попытка получить общий объем VRAM
                try:
                    result = self.ssh_client.execute(
                        "nvidia-smi --query-gpu=memory.total --format=csv,nounits,noheader",
                        timeout=10
                    )
                    if result.returncode == 0:
                        vram_total = int(result.stdout.strip()) * 1024 * 1024
                        vram_percent = (vram_used / vram_total) * 100 if vram_total > 0 else 0
                except Exception:
                    pass
            
            timestamp = time.time()
            
            stats = SystemStats(
                cpu_percent=cpu_percent,
                ram_used=ram_used,
                ram_total=ram_total,
                ram_percent=ram_percent,
                vram_used=vram_used,
                vram_total=vram_total,
                vram_percent=vram_percent,
                timestamp=timestamp
            )
            
            # Добавляем выборки для агрегации
            stats.add_sample(cpu_percent, ram_used, vram_used)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error collecting remote system stats: {e}")
            return None
    
    def _get_cpu_usage_remote(self) -> float:
        """Получить загрузку CPU на удаленной машине."""
        try:
            # Используем top для получения загрузки CPU
            cmd = "top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1"
            result = self.ssh_client.execute(cmd, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception as e:
            logger.debug(f"Error getting remote CPU usage: {e}")
        return 0.0
    
    def _get_ram_usage_remote(self) -> dict[str, Any] | None:
        """Получить статистику RAM на удаленной машине."""
        try:
            # Используем free для получения статистики RAM
            cmd = "free -m | grep '^Mem:' | awk '{print $3,$2,$7}'"
            result = self.ssh_client.execute(cmd, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                used_mb, total_mb, available_mb = map(int, result.stdout.strip().split())
                return {
                    'used': used_mb * 1024 * 1024,  # Convert to bytes
                    'total': total_mb * 1024 * 1024,
                    'available': available_mb * 1024 * 1024,
                    'percent': (used_mb / total_mb) * 100 if total_mb > 0 else 0.0
                }
        except Exception as e:
            logger.debug(f"Error getting remote RAM usage: {e}")
        return None


class ResourceMonitor:
    """Универсальный интерфейс для мониторинга ресурсов."""
    
    def __init__(self, monitor_type: str = 'local', collection_interval: float = 0.5,
                 ssh_client: Any = None) -> None:
        """Инициализация монитора ресурсов.
        
        Args:
            monitor_type: Тип монитора ('local' или 'remote')
            collection_interval: Интервал сбора метрик в секундах
            ssh_client: SSHClient для удаленного мониторинга
        """
        self.monitor_type = monitor_type
        self.collection_interval = collection_interval
        self.monitor: BaseMonitor | None = None
        
        if monitor_type == 'local':
            self.monitor = LocalSystemMonitor(collection_interval)
        elif monitor_type == 'remote' and ssh_client:
            self.monitor = RemoteSystemMonitor(ssh_client, collection_interval)
        else:
            raise ValueError(f"Unsupported monitor type: {monitor_type}")
    
    def start_monitoring(self, duration: float | None = None) -> None:
        """Начать мониторинг."""
        if self.monitor:
            self.monitor.start_monitoring(duration)
    
    def stop_monitoring(self) -> None:
        """Остановить мониторинг."""
        if self.monitor:
            self.monitor.stop_monitoring()
    
    def get_latest_stats(self) -> SystemStats | None:
        """Получить последние метрики."""
        return self.monitor.get_latest_stats() if self.monitor else None
    
    def get_aggregated_stats(self) -> dict[str, Any] | None:
        """Получить агрегированную статистику."""
        return self.monitor.get_aggregated_stats() if self.monitor else None
    
    def get_stats_history(self) -> list[SystemStats]:
        """Получить историю метрик."""
        return self.monitor.get_stats_history() if self.monitor else []
    
    def is_monitoring(self) -> bool:
        """Проверяет, идет ли мониторинг."""
        return self.monitor.is_monitoring if self.monitor else False


# Утилитарные функции для быстрого использования
def get_local_stats(collection_interval: float = 0.5, samples: int = 1) -> dict[str, Any] | None:
    """Получить метрики локальной системы с заданным количеством выборок.
    
    Args:
        collection_interval: Интервал между выборками в секундах
        samples: Количество выборок для сбора
    
    Returns:
        SystemStats с агрегированными метриками
    """
    monitor = LocalSystemMonitor(collection_interval)
    monitor.start_monitoring(duration=collection_interval * samples)
    
    # Ждем завершения сбора
    time.sleep(collection_interval * samples + 0.1)
    monitor.stop_monitoring()
    
    return monitor.get_aggregated_stats()


def get_remote_stats(ssh_client: Any, collection_interval: float = 0.5, samples: int = 1) -> dict[str, Any] | None:
    """Получить метрики удаленной системы с заданным количеством выборок.
    
    Args:
        ssh_client: SSHClient для удаленного доступа
        collection_interval: Интервал между выборками в секундах
        samples: Количество выборок для сбора
    
    Returns:
        SystemStats с агрегированными метриками
    """
    monitor = RemoteSystemMonitor(ssh_client, collection_interval)
    monitor.start_monitoring(duration=collection_interval * samples)
    
    # Ждем завершения сбора
    time.sleep(collection_interval * samples + 0.1)
    monitor.stop_monitoring()
    
    return monitor.get_aggregated_stats()