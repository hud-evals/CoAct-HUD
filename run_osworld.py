#!/usr/bin/env python3
"""
Quick runner script for OSWorld-Verified-XLang taskset using CoAct agent.
"""
import sys
import os

# Add current directory to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coact_taskset_runner import main  # type: ignore

if __name__ == "__main__":
    # Add default taskset if not specified
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and not any(arg.startswith('--taskset') for arg in sys.argv)):
        sys.argv.extend(['--taskset', 'OSWorld-Verified-XLang'])
    
    # Add default name if not specified
    if not any(arg.startswith('--name') for arg in sys.argv):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sys.argv.extend(['--name', f'OSWorld_CoAct_{timestamp}'])
    
    # Run with provided or default arguments
    sys.exit(main())
