#!/usr/bin/env python3
"""Generate SHA256 checksums for run artifacts"""

import sys
import hashlib
import json
from pathlib import Path

def generate_sha256(filepath):
    """Generate SHA256 hash for a file"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        print(f"Error hashing {filepath}: {e}")
        return None

def generate_checksums(run_id):
    """Generate checksums for all run artifacts"""
    run_dir = Path(f"data/runs/{run_id}")
    
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return False
    
    checksums = {}
    
    # Original artifacts
    original_files = ['manifest.json', 'metrics.json', 'trades.csv', 'events.csv', 'series.csv']
    for filename in original_files:
        filepath = run_dir / filename
        if filepath.exists():
            checksum = generate_sha256(filepath)
            if checksum:
                checksums[f"original_{filename}"] = checksum
    
    # Analysis artifacts
    analysis_dir = run_dir / "analysis"
    if analysis_dir.exists():
        for filepath in analysis_dir.glob("*.json"):
            checksum = generate_sha256(filepath)
            if checksum:
                checksums[f"analysis_{filepath.name}"] = checksum
    
    # Visualization artifacts (key files only)
    figs_dir = run_dir / "figs" 
    if figs_dir.exists():
        key_vis_files = ['main_analysis.png', 'main_analysis.pdf', 'trade_analysis.png', 'performance_dashboard.png']
        for filename in key_vis_files:
            filepath = figs_dir / filename
            if filepath.exists():
                checksum = generate_sha256(filepath)
                if checksum:
                    checksums[f"visualization_{filename}"] = checksum
    
    # Save checksums
    checksum_file = run_dir / "checksums.json"
    with open(checksum_file, 'w') as f:
        json.dump(checksums, f, indent=2)
    
    print(f"Generated checksums for {len(checksums)} files")
    print(f"Checksums saved to: {checksum_file}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_checksums.py <run_id>")
        sys.exit(1)
    
    run_id = sys.argv[1]
    success = generate_checksums(run_id)
    sys.exit(0 if success else 1)