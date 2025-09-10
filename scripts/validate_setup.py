#!/usr/bin/env python3
"""
System Setup Validation Script
Validates all dependencies and system requirements for strategy development
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import importlib.util

def check_python_version():
    """Check Python version meets requirements"""
    version = sys.version_info
    min_version = (3, 11)
    
    if version >= min_version:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor} < required 3.11"

def check_package(package_name, min_version=None):
    """Check if a package is installed and meets version requirements"""
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return False, f"{package_name} not installed"
    
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            version = module.__version__
            if min_version and version < min_version:
                return False, f"{package_name} {version} < required {min_version}"
            return True, f"{package_name} {version}"
        return True, f"{package_name} installed"
    except Exception as e:
        return False, f"{package_name} import failed: {e}"

def check_git():
    """Check if git is available"""
    try:
        result = subprocess.run(['git', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, "Git command failed"
    except FileNotFoundError:
        return False, "Git not found"

def check_directory_structure():
    """Check required directory structure exists"""
    required_dirs = [
        'cloud/tasks',
        'cloud/state',
        'docs',
        'docs/runs',
        'docs/guides',
        'scripts',
        'tools',
        'tools/latex'
    ]
    
    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(dir_path)
    
    if missing:
        return False, f"Missing directories: {', '.join(missing)}"
    return True, "All required directories present"

def check_documentation():
    """Check if key documentation files exist"""
    required_docs = [
        'docs/EMR.md',
        'docs/SMR.md',
        'docs/ECL.md',
        'docs/SCL.md',
        'docs/guides/STRAT_TEMPLATE.md'
    ]
    
    missing = []
    for doc_path in required_docs:
        if not Path(doc_path).exists():
            missing.append(doc_path)
    
    if missing:
        return False, f"Missing documents: {', '.join(missing)}"
    return True, "All documentation files present"

def main():
    """Run all validation checks"""
    print("=" * 60)
    print("SYSTEM SETUP VALIDATION")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Define all checks
    checks = [
        ("Python Version", check_python_version),
        ("Git", check_git),
        ("Directory Structure", check_directory_structure),
        ("Documentation", check_documentation),
        ("pandas", lambda: check_package('pandas')),
        ("numpy", lambda: check_package('numpy')),
        ("matplotlib", lambda: check_package('matplotlib')),
        ("seaborn", lambda: check_package('seaborn')),
        ("plotly", lambda: check_package('plotly')),
        ("scipy", lambda: check_package('scipy')),
        ("scikit-learn", lambda: check_package('sklearn')),
        ("ccxt", lambda: check_package('ccxt')),
        ("tqdm", lambda: check_package('tqdm')),
        ("numba", lambda: check_package('numba')),
    ]
    
    results = {}
    all_passed = True
    
    print("Running validation checks...")
    print("-" * 60)
    
    for check_name, check_func in checks:
        try:
            passed, message = check_func()
            status = "PASSED" if passed else "FAILED"
            print(f"{check_name:20} {status:10} {message}")
            results[check_name] = {
                "passed": passed,
                "message": message
            }
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"{check_name:20} ERROR      {str(e)}")
            results[check_name] = {
                "passed": False,
                "message": f"Error: {str(e)}"
            }
            all_passed = False
    
    print("-" * 60)
    
    # Summary
    passed_count = sum(1 for r in results.values() if r["passed"])
    total_count = len(results)
    
    print()
    print("VALIDATION SUMMARY")
    print("-" * 60)
    print(f"Checks Passed: {passed_count}/{total_count}")
    print(f"Overall Status: {'ALL CHECKS PASSED' if all_passed else 'VALIDATION FAILED'}")
    
    # Write results to file
    output_file = Path('cloud/state/validation_results.json')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    validation_data = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "passed" if all_passed else "failed",
        "passed_count": passed_count,
        "total_count": total_count,
        "checks": results
    }
    
    with open(output_file, 'w') as f:
        json.dump(validation_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Exit code
    if all_passed:
        print("\n[SUCCESS] System ready for strategy development!")
        return 0
    else:
        print("\n[FAILURE] Please address failed checks before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())