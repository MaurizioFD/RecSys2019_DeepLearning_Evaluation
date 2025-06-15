"""
Simple validation test to ensure coverage reporting works.
"""


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


def divide_numbers(a: int, b: int) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


class TestSimpleValidation:
    """Test basic functionality to validate coverage."""
    
    def test_add_numbers(self):
        """Test addition function."""
        assert add_numbers(2, 3) == 5
        assert add_numbers(-1, 1) == 0
        assert add_numbers(0, 0) == 0
    
    def test_multiply_numbers(self):
        """Test multiplication function."""
        assert multiply_numbers(2, 3) == 6
        assert multiply_numbers(-2, 3) == -6
        assert multiply_numbers(0, 100) == 0