# Test Example Module

import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.example import hello_world

def test_hello_world():
    """Test the hello_world function."""
    result = hello_world()
    assert result == "Hello, World!"
    print("✓ test_hello_world passed")

if __name__ == "__main__":
    test_hello_world()
    print("\nAll tests passed!")
