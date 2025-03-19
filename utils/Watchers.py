import time
import subprocess
from .meassurator import GPUWatcher, CPUWatcher

class ExperimentWatcher:
    def __init__(self, base_path: str):
        self.base_path = base_path

    def measure_energy_consumption(self, executable_file_path: str, timelapse: int = 1, split = "balanced", subinfosize = 6):
        """Ejecuta un script y mide consumo de CPU/GPU cada cierto tiempo."""
        proc = subprocess.Popen(['python3', executable_file_path, split, str(subinfosize)])
        gpu_watcher = GPUWatcher(pids=[proc.pid], base_path=self.base_path)
        cpu_watcher = CPUWatcher(base_path=self.base_path)
        while proc.poll() is None:
            # Procesar o almacenar gpu_data y cpu_data
            gpu_watcher.record()
            cpu_watcher.record()
            time.sleep(timelapse)
        gpu_watcher.save()
        cpu_watcher.save()
        return proc.returncode

class ReplicationWatcher:
    
    def __init__(self, base_path: str):
        self.base_path = base_path

    def measure_energy_consumption(self, cmd, timelapse = 1):
        """Ejecuta un script y mide consumo de CPU/GPU cada cierto tiempo."""
        start_time = time.time() 

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        gpu_watcher = GPUWatcher(pids=[proc.pid], base_path=self.base_path)
        cpu_watcher = CPUWatcher(base_path=self.base_path)
        while proc.poll() is None:
            # Procesar o almacenar gpu_data y cpu_data
            gpu_watcher.record()
            cpu_watcher.record()
            time.sleep(timelapse)
        gpu_watcher.save()
        cpu_watcher.save()

        output, _ = proc.communicate()
        elapsed_time = time.time() - start_time
        return output, elapsed_time