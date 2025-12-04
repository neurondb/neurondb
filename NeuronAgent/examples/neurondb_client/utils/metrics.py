"""
Metrics collection utilities
"""

import time
from typing import Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Metric:
    """Single metric value"""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collect and aggregate metrics
    
    Usage:
        collector = MetricsCollector()
        collector.record("request_duration", 0.5)
        collector.record("tokens_used", 100)
        metrics = collector.get_summary()
    """
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics: List[Metric] = []
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)
    
    def record(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """
        Record a metric
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags
        """
        metric = Metric(name, value, tags=tags or {})
        self.metrics.append(metric)
    
    def increment(self, name: str, value: int = 1) -> None:
        """
        Increment a counter
        
        Args:
            name: Counter name
            value: Increment value
        """
        self.counters[name] += value
    
    def timer(self, name: str, duration: float) -> None:
        """
        Record a timer value
        
        Args:
            name: Timer name
            duration: Duration in seconds
        """
        self.timers[name].append(duration)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary
        
        Returns:
            Dictionary with aggregated metrics
        """
        summary = {
            'counters': dict(self.counters),
            'timers': {}
        }
        
        for name, values in self.timers.items():
            if values:
                summary['timers'][name] = {
                    'count': len(values),
                    'total': sum(values),
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        
        return summary
    
    def reset(self) -> None:
        """Reset all metrics"""
        self.metrics.clear()
        self.counters.clear()
        self.timers.clear()

