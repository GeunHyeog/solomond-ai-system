#!/usr/bin/env python3
"""
Simple Performance Monitor for Analysis Testing
Real-time monitoring of file analysis performance
"""

import time
import json
from datetime import datetime

class SimplePerformanceMonitor:
    """Simple performance monitoring without unicode issues"""
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = []
        
    def start(self, task_name="Analysis"):
        """Start monitoring"""
        self.start_time = time.time()
        self.task_name = task_name
        
        print(f"=== {task_name} Started ===")
        print(f"Start Time: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 40)
        
    def checkpoint(self, step_name):
        """Log checkpoint"""
        if not self.start_time:
            print("Error: Monitoring not started")
            return
            
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        checkpoint_data = {
            "step": step_name,
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60,
            "timestamp": datetime.now().strftime('%H:%M:%S')
        }
        
        self.checkpoints.append(checkpoint_data)
        
        print(f"STEP: {step_name}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"  Clock: {checkpoint_data['timestamp']}")
        
    def finish(self):
        """Finish monitoring"""
        if not self.start_time:
            print("Error: Monitoring not started")
            return None
            
        total_time = time.time() - self.start_time
        
        result = {
            "task_name": self.task_name,
            "total_seconds": total_time,
            "total_minutes": total_time / 60,
            "checkpoints": self.checkpoints,
            "completed_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print("-" * 40)
        print(f"=== {self.task_name} Completed ===")
        print(f"Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Total Steps: {len(self.checkpoints)}")
        
        # Save log
        log_file = f"analysis_timing_{datetime.now().strftime('%H%M%S')}.json"
        with open(log_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Log saved: {log_file}")
        
        return result
    
    def current_time(self):
        """Get current elapsed time"""
        if not self.start_time:
            return 0
        return time.time() - self.start_time

# Global monitor instance
monitor = SimplePerformanceMonitor()

def start_timing(task_name="File Analysis"):
    """Start timing"""
    monitor.start(task_name)

def log_step(step_name):
    """Log analysis step"""
    monitor.checkpoint(step_name)

def finish_timing():
    """Finish timing"""
    return monitor.finish()

def current_elapsed():
    """Get current elapsed time"""
    return monitor.current_time()

if __name__ == "__main__":
    # Test
    print("Testing Performance Monitor...")
    
    start_timing("Test Analysis")
    
    import time
    log_step("File Loading")
    time.sleep(1)
    
    log_step("Processing")
    time.sleep(2)
    
    log_step("AI Analysis")
    time.sleep(3)
    
    result = finish_timing()
    print(f"Test completed in {result['total_seconds']:.1f} seconds")