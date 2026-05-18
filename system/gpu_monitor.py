"""GPU detection and VRAM monitoring utilities."""

import subprocess
import os
import time


def check_gpu_available() -> bool:
    """Check if NVIDIA GPU is available.

    Returns:
        bool: True if GPU is available, False otherwise
    """
    # Check for nvidia-smi availability
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return True
    except Exception:
        pass

    # Check for /proc/driver/nvidia/gpus/
    try:
        return os.path.exists('/proc/driver/nvidia/gpus/')
    except Exception:
        pass

    return False


def get_vram_usage() -> int | None:
    """Get current VRAM usage in bytes.

    Returns:
        int or None: VRAM usage in bytes, or None if GPU is unavailable
    """
    if not check_gpu_available():
        return None

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            # nvidia-smi returns value in MiB, convert to bytes
            return int(result.stdout.strip()) * 1024 * 1024
    except Exception:
        pass

    # Fallback: read from /proc/driver/nvidia/gpus/0/mem_used
    try:
        gpu_path = '/proc/driver/nvidia/gpus/0/mem_used'
        if os.path.exists(gpu_path):
            with open(gpu_path, 'r') as f:
                content = f.read().strip()
                # Format: "used: 4500MiB" or "4500 MiB"
                if ':' in content:
                    value = content.split(':')[1].strip()
                else:
                    value = content
                # Parse value with MiB/GiB
                value = value.strip()
                if value.endswith('GiB'):
                    return int(value[:-3]) * 1024 * 1024 * 1024
                elif value.endswith('MiB'):
                    return int(value[:-3]) * 1024 * 1024
                elif value.endswith('GB'):
                    return int(value[:-2]) * 1024 * 1024 * 1024
                elif value.endswith('MB'):
                    return int(value[:-2]) * 1024 * 1024
                else:
                    return int(value)
    except Exception:
        pass

    return None


def get_vram_stats(samples: int = 10, interval: float = 0.5) -> dict:
    """Get comprehensive VRAM statistics with min/max/avg values.

    Args:
        samples: Number of samples to collect
        interval: Interval between samples in seconds

    Returns:
        dict: VRAM statistics including current, min, max, avg, total
              Returns None if GPU is unavailable
    """
    if not check_gpu_available():
        return None

    usages = []
    total_vram = None

    # Collect samples
    for _ in range(samples):
        usage = get_vram_usage()
        if usage is not None:
            usages.append(usage)
        time.sleep(interval)

    if not usages:
        return None

    # Get total VRAM
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            total_vram = int(result.stdout.strip()) * 1024 * 1024
    except Exception:
        # Fallback: use max observed usage as estimate
        total_vram = max(usages) * 1.2  # Rough estimate

    # Calculate statistics
    current_usage = usages[-1]
    min_usage = min(usages)
    max_usage = max(usages)
    avg_usage = sum(usages) / len(usages)

    return {
        'current': current_usage,
        'min': min_usage,
        'max': max_usage,
        'avg': avg_usage,
        'total': total_vram,
        'percent_current': (current_usage / total_vram) * 100 if total_vram > 0 else 0,
        'percent_min': (min_usage / total_vram) * 100 if total_vram > 0 else 0,
        'percent_max': (max_usage / total_vram) * 100 if total_vram > 0 else 0,
        'percent_avg': (avg_usage / total_vram) * 100 if total_vram > 0 else 0,
        'samples_count': len(usages),
        'interval': interval
    }


def get_vram_usage_history(samples: int = 10, interval: float = 0.5) -> list:
    """Get VRAM usage history over time.

    Args:
        samples: Number of samples to collect
        interval: Interval between samples in seconds

    Returns:
        list: List of VRAM usage values in bytes
              Returns empty list if GPU is unavailable
    """
    if not check_gpu_available():
        return []

    usages = []
    for _ in range(samples):
        usage = get_vram_usage()
        if usage is not None:
            usages.append(usage)
        time.sleep(interval)

    return usages


def get_vram_total() -> int | None:
    """Get total VRAM capacity.

    Returns:
        int or None: Total VRAM in bytes, or None if GPU is unavailable
    """
    if not check_gpu_available():
        return None

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip()) * 1024 * 1024
    except Exception:
        pass

    return None


def get_gpu_utilization() -> float | None:
    """Get current GPU utilization percentage.

    Returns:
        float or None: GPU utilization percentage (0-100), or None if GPU is unavailable
    """
    if not check_gpu_available():
        return None

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass

    return None


def get_gpu_stats(samples: int = 10, interval: float = 0.5) -> dict:
    """Get comprehensive GPU statistics with min/max/avg values.

    Args:
        samples: Number of samples to collect
        interval: Interval between samples in seconds

    Returns:
        dict: GPU statistics including utilization and VRAM
        Returns None if GPU is unavailable
    """
    if not check_gpu_available():
        return None

    utilizations = []
    vram_usages = []

    for _ in range(samples):
        util = get_gpu_utilization()
        vram = get_vram_usage()
        if util is not None:
            utilizations.append(util)
        if vram is not None:
            vram_usages.append(vram)
        time.sleep(interval)

    if not utilizations:
        return None

    total_vram = get_vram_total()

    return {
        'utilization_current': utilizations[-1] if utilizations else 0,
        'utilization_min': min(utilizations) if utilizations else 0,
        'utilization_max': max(utilizations) if utilizations else 0,
        'utilization_avg': sum(utilizations) / len(utilizations) if utilizations else 0,
        'vram_current': vram_usages[-1] if vram_usages else None,
        'vram_min': min(vram_usages) if vram_usages else None,
        'vram_max': max(vram_usages) if vram_usages else None,
        'vram_avg': sum(vram_usages) / len(vram_usages) if vram_usages else None,
        'vram_total': total_vram,
        'samples_count': len(utilizations),
        'interval': interval
    }
