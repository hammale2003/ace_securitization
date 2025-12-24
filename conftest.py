"""
Pytest configuration and fixtures for ACE Securitization tests.
"""
import sys
from pathlib import Path

# Add the current directory to Python path for imports
test_dir = Path(__file__).parent
if str(test_dir) not in sys.path:
    sys.path.insert(0, str(test_dir))

