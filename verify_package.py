#!/usr/bin/env python3
"""
Verify that the BOMAX package structure is correct and that all modules can be imported.
"""

import sys
import importlib

def check_import(module_name):
    try:
        module = importlib.import_module(module_name)
        print(f"✓ Successfully imported {module_name}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {module_name}: {e}")
        return False

def main():
    # List of modules to check
    modules = [
        "bomax",
        "bomax.initialize",
        "bomax.sampler",
        "bomax.normalize",
        "bomax.degree",
        "bomax.stopping",
        "bomax.utils",
    ]
    
    # Check each module
    success = True
    for module in modules:
        if not check_import(module):
            success = False
    
    # Print summary
    if success:
        print("\nAll modules imported successfully!")
        print("The package structure is correct.")
    else:
        print("\nSome modules failed to import.")
        print("Make sure you have installed the package in development mode:")
        print("  pip install -e .")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
