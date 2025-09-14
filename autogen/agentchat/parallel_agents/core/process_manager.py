# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import os

from ..utils.core_affinity import CoreAffinityManager


def core_worker_process(core_id: int, input_queue: mp.Queue, output_queue: mp.Queue):
    """Worker function that runs in each core process"""
    # Bind this process to the specific core
    affinity_manager = CoreAffinityManager()
    affinity_manager.bind_current_process_to_core(core_id)

    print(f"Core worker {core_id} started and bound to CPU core {core_id}")

    while True:
        try:
            # Get task from input queue
            task = input_queue.get(timeout=1.0)

            if task is None:  # Shutdown signal
                break

            # Process the task
            result = {
                "core_id": core_id,
                "task_id": task.get("task_id", "unknown"),
                "result": f"Processed on core {core_id}: {task}",
                "status": "completed",
            }

            # Send result to output queue
            output_queue.put(result)

        except mp.queues.Empty:
            continue
        except Exception as e:
            error_result = {"core_id": core_id, "error": str(e), "status": "error"}
            output_queue.put(error_result)


class ProcessManager:
    """Manages worker processes for each CPU core"""

    def __init__(self, num_cores: int = None):
        self.num_cores = num_cores or os.cpu_count()
        self.core_processes: dict[int, mp.Process] = {}
        self.core_queues: dict[int, dict[str, mp.Queue]] = {}
        self.affinity_manager = CoreAffinityManager()
        self._task_counter = 0  # Simple counter to avoid using qsize()

    def spawn_core_worker(self, core_id: int) -> mp.Process:
        """Spawn process and bind to core using CoreAffinityManager"""
        # Create input and output queues for this core
        input_queue = mp.Queue()
        output_queue = mp.Queue()

        # Store queues
        self.core_queues[core_id] = {"input": input_queue, "output": output_queue}

        # Create and start the worker process
        process = mp.Process(
            target=core_worker_process, args=(core_id, input_queue, output_queue), name=f"CoreWorker-{core_id}"
        )

        process.start()
        self.core_processes[core_id] = process

        return process

    def distribute_task(self, task_data: dict, target_core: int):
        """Put task in core's queue"""
        if target_core not in self.core_queues:
            raise ValueError(f"Core {target_core} not initialized")

        # Add task ID if not present - use counter instead of qsize()
        if "task_id" not in task_data:
            self._task_counter += 1
            task_data["task_id"] = f"task_{self._task_counter}"

        # Send task to the target core
        self.core_queues[target_core]["input"].put(task_data)

    def initialize_cores(self):
        """Call spawn_core_worker for each core"""
        print(f"Initializing {self.num_cores} core workers...")

        for core_id in range(self.num_cores):
            self.spawn_core_worker(core_id)

        print(f"All {self.num_cores} core workers initialized")

    def get_result(self, core_id: int, timeout: float = 5.0):
        """Get result from specific core"""
        if core_id not in self.core_queues:
            return None

        try:
            return self.core_queues[core_id]["output"].get(timeout=timeout)
        except mp.queues.Empty:
            return None

    def shutdown(self):
        """Shutdown all core workers"""
        print("Shutting down core workers...")

        # Send shutdown signal to all cores
        for core_id in self.core_queues:
            self.core_queues[core_id]["input"].put(None)

        # Wait for all processes to finish
        for process in self.core_processes.values():
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()

        print("All core workers shutdown")
