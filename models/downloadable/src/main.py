"""
Downloadable NIM Runner for Dataloop.

Starts the NIM inference server and streams logs with GPU memory monitoring.
"""

import logging
import subprocess
import threading
import time

import dtlpy as dl

# Configure logging - NIM server output will also use this logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("NIM-Runner")

SERVER_SCRIPT = "/opt/nim/start_server.sh"
GPU_LOG_INTERVAL = 60  # seconds between GPU memory logs


def _stream_output(pipe, log_level=logging.INFO, prefix=""):
    """Stream subprocess output to logger."""
    try:
        for line in iter(pipe.readline, ""):
            if line:
                msg = line.rstrip('\n\r')
                if prefix:
                    msg = f"{prefix}{msg}"
                logger.log(log_level, msg)
    finally:
        pipe.close()


def _get_gpu_memory():
    """Get GPU memory stats using nvidia-smi.
    
    Returns:
        Tuple of (free, total, used) lists in MB, or None if nvidia-smi fails.
    """
    try:
        # Get free memory
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False
        )
        free = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
        
        # Get total memory
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False
        )
        total = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
        
        # Get used memory
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=False
        )
        used = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
        
        return free, total, used
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


class Runner(dl.BaseServiceRunner):
    """
    NIM runner that starts the inference server and streams logs.
    
    The server exposes an OpenAI-compatible API on port 3000.
    GPU memory is logged periodically for monitoring.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        logger.info("Starting NIM inference server...")
        
        # Start the NIM server
        self.server_process = subprocess.Popen(
            [SERVER_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        
        # Stream stdout and stderr to logger
        threading.Thread(
            target=_stream_output,
            args=(self.server_process.stdout, logging.INFO),
            daemon=True,
        ).start()
        
        threading.Thread(
            target=_stream_output,
            args=(self.server_process.stderr, logging.WARNING, "[stderr] "),
            daemon=True,
        ).start()
        
        # Start GPU monitoring
        threading.Thread(target=self._monitor_gpu, daemon=True).start()
        
        logger.info("NIM server started, streaming logs...")

    def _monitor_gpu(self):
        """Log GPU memory usage periodically."""
        while True:
            stats = _get_gpu_memory()
            if stats:
                free, total, used = stats
                logger.info(f"GPU memory - total: {total} MB, used: {used} MB, free: {free} MB")
            time.sleep(GPU_LOG_INTERVAL)


if __name__ == "__main__":
    runner = Runner()
    runner.server_process.wait()
