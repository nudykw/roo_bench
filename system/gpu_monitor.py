"""GPU detection and VRAM monitoring utilities."""

import subprocess
import os


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
            return int(result.stdout.strip())
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
