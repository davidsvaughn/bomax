#!/usr/bin/env python3
"""
Test script to verify that the bomax package can be installed and imported.
"""

try:
    import bomax
    print(f"Successfully imported bomax version {bomax.__version__}")
    print("The following modules are available:")
    for module in dir(bomax):
        if not module.startswith('__'):
            print(f"  - {module}")
except ImportError as e:
    print(f"Failed to import bomax: {e}")
    print("Make sure to install the package in editable mode with:")
    print("  pip install -e .")
