#!/usr/bin/env python3
"""Update run registry with analysis status"""

import sys
import csv
import time
import os
from pathlib import Path
from datetime import datetime, timezone

def update_run_registry(run_id, analysis_status="analyzed"):
    """Update run registry with analysis completion status"""
    
    registry_file = Path("docs/runs/run_registry.csv")
    lockfile = Path("docs/runs/.registry.lock")
    
    # Check for stale lockfile
    if lockfile.exists():
        try:
            stat = lockfile.stat()
            age_seconds = time.time() - stat.st_mtime
            if age_seconds > 300:  # 5 minutes
                print(f"Warning: Stale lockfile detected ({age_seconds:.1f}s old), removing...")
                lockfile.unlink()
        except Exception as e:
            print(f"Warning: Could not check lockfile age: {e}")
    
    # Create lockfile
    try:
        with open(lockfile, 'w') as f:
            f.write(f"PID: {os.getpid()}\nTimestamp: {datetime.now(timezone.utc).isoformat()}\n")
    except Exception as e:
        print(f"Warning: Could not create lockfile: {e}")
    
    try:
        # Read current registry
        rows = []
        headers = []
        
        if registry_file.exists():
            with open(registry_file, 'r', newline='') as f:
                reader = csv.reader(f)
                headers = next(reader, [])
                rows = list(reader)
        
        # Check if we need to add analysis_status column
        if 'analysis_status' not in headers:
            headers.append('analysis_status')
            headers.append('analysis_timestamp')
            # Add empty values for existing rows
            for row in rows:
                while len(row) < len(headers):
                    row.append('')
        
        # Find and update the target run
        run_found = False
        analysis_col = headers.index('analysis_status') if 'analysis_status' in headers else len(headers)
        timestamp_col = headers.index('analysis_timestamp') if 'analysis_timestamp' in headers else len(headers)
        
        for row in rows:
            if len(row) > 0 and row[0] == run_id:  # Assuming run_id is first column
                # Extend row if needed
                while len(row) < len(headers):
                    row.append('')
                    
                row[analysis_col] = analysis_status
                row[timestamp_col] = datetime.now(timezone.utc).isoformat()
                run_found = True
                break
        
        if not run_found:
            print(f"Warning: Run {run_id} not found in registry")
            return False
        
        # Write updated registry
        with open(registry_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        
        print(f"Successfully updated registry: {run_id} -> {analysis_status}")
        return True
        
    except Exception as e:
        print(f"Error updating registry: {e}")
        return False
        
    finally:
        # Remove lockfile
        try:
            if lockfile.exists():
                lockfile.unlink()
        except Exception as e:
            print(f"Warning: Could not remove lockfile: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python update_registry.py <run_id>")
        sys.exit(1)
    
    run_id = sys.argv[1]
    success = update_run_registry(run_id)
    sys.exit(0 if success else 1)