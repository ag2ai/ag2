# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
import platform

import psutil

__all__ = ["CoreAffinityManager"]


class CoreAffinityManager:
    """Manages CPU core affinity for processes"""

    def __init__(self):
        self.available_cores = list(range(os.cpu_count()))
        self.platform = platform.system().lower()
        self._supports_cpu_affinity = self._check_cpu_affinity_support()

    def _check_cpu_affinity_support(self) -> bool:
        """Check if the current platform supports cpu_affinity"""
        return self.platform == "linux"

    def bind_current_process_to_core(self, core_id: int) -> bool:
        """Bind current process to specific CPU core using psutil (Linux only)"""
        if not self._supports_cpu_affinity:
            print(f"CPU affinity not supported on {self.platform}. Skipping core binding.")
            return True  # Return True to indicate "success" even though we can't bind

        try:
            if core_id not in self.available_cores:
                return False

            current_process = psutil.Process()
            current_process.cpu_affinity([core_id])
            print(f"Successfully bound process to core {core_id}")
            return True
        except (psutil.Error, OSError, AttributeError) as e:
            print(f"Failed to bind process to core {core_id}: {e}")
            return False

    def get_available_cores(self) -> list[int]:
        """Return list of available CPU cores"""
        return list(range(os.cpu_count()))

    def distribute_processes_across_cores(self, process_ids: list[int]) -> dict[int, int]:
        """Simple round-robin assignment of processes to cores"""
        distribution = {}
        num_cores = len(self.available_cores)

        for i, process_id in enumerate(process_ids):
            core_id = self.available_cores[i % num_cores]
            distribution[process_id] = core_id

        return distribution
