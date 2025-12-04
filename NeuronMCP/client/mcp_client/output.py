"""
Output management for command results.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


class OutputManager:
    """Manages output file generation for command results."""

    def __init__(self, output_path: Optional[str] = None):
        """
        Initialize output manager.

        Args:
            output_path: Optional output file path. If None, generates timestamped filename.
        """
        self.output_path = output_path
        self.results: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

    def add_result(self, command: str, result: Any):
        """
        Add a command result.

        Args:
            command: Command that was executed
            result: Result from command execution
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "result": result,
        }
        self.results.append(entry)

    def save(self) -> str:
        """
        Save results to file.

        Returns:
            Path to saved file
        """
        if self.output_path:
            output_file = Path(self.output_path)
        else:
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            output_file = Path(f"results_{timestamp}.json")

        # Ensure directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Prepare output data
        output_data = {
            "metadata": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_commands": len(self.results),
                "successful_commands": sum(1 for r in self.results if "error" not in r.get("result", {})),
                "failed_commands": sum(1 for r in self.results if "error" in r.get("result", {})),
            },
            "results": self.results,
        }

        # Write to file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        return str(output_file)

