"""Tests for CLI parsing functions."""

import unittest

from cli import parse_context_size


class TestParseContextSize(unittest.TestCase):
    """Tests for parse_context_size function."""

    def test_plain_number(self):
        """Test parsing plain numbers."""
        self.assertEqual(parse_context_size("131072"), 131072)
        self.assertEqual(parse_context_size("2048"), 2048)
        self.assertEqual(parse_context_size("8192"), 8192)

    def test_k_suffix_lowercase(self):
        """Test parsing K suffix (lowercase)."""
        self.assertEqual(parse_context_size("8k"), 8192)
        self.assertEqual(parse_context_size("16k"), 16384)
        self.assertEqual(parse_context_size("128k"), 131072)
        self.assertEqual(parse_context_size("2048k"), 2097152)

    def test_k_suffix_uppercase(self):
        """Test parsing K suffix (uppercase)."""
        self.assertEqual(parse_context_size("8K"), 8192)
        self.assertEqual(parse_context_size("16K"), 16384)
        self.assertEqual(parse_context_size("128K"), 131072)
        self.assertEqual(parse_context_size("2048K"), 2097152)

    def test_k_suffix_mixed_case(self):
        """Test parsing K suffix (mixed case)."""
        self.assertEqual(parse_context_size("8K"), 8192)
        self.assertEqual(parse_context_size("128K"), 131072)

    def test_m_suffix(self):
        """Test parsing M suffix."""
        self.assertEqual(parse_context_size("1M"), 1048576)
        self.assertEqual(parse_context_size("2M"), 2097152)

    def test_with_spaces(self):
        """Test parsing with leading/trailing spaces."""
        self.assertEqual(parse_context_size("  128K  "), 131072)
        self.assertEqual(parse_context_size("  16K"), 16384)

    def test_float_values(self):
        """Test parsing float values with K suffix."""
        self.assertEqual(parse_context_size("0.5K"), 512)
        self.assertEqual(parse_context_size("1.5K"), 1536)
        self.assertEqual(parse_context_size("2.5K"), 2560)

    def test_invalid_input(self):
        """Test parsing invalid input raises ValueError."""
        with self.assertRaises(ValueError):
            parse_context_size("invalid")
        with self.assertRaises(ValueError):
            parse_context_size("abcK")
        with self.assertRaises(ValueError):
            parse_context_size("")


if __name__ == "__main__":
    unittest.main()