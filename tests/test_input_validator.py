"""Tests for input validation module."""

from unittest.mock import patch

import pytest

from ui.input_validator import InputValidator


class TestPromptYesNo:
    """Tests for prompt_yes_no method."""

    @pytest.mark.parametrize("user_input", ['y', 'Y'])
    def test_yes_values(self, user_input):
        """Test that 'y' returns True."""
        with patch('builtins.input', return_value=user_input):
            result = InputValidator.prompt_yes_no("Test question")
            assert result is True

    @pytest.mark.parametrize("user_input", ['n', 'N'])
    def test_no_values(self, user_input):
        """Test that 'n' returns False."""
        with patch('builtins.input', return_value=user_input):
            result = InputValidator.prompt_yes_no("Test question")
            assert result is False

    def test_invalid_input_then_valid(self):
        """Test that invalid input triggers retry and then accepts valid input."""
        with patch('builtins.input', side_effect=['invalid', 'yes', 'y']):
            with patch('builtins.print') as mock_print:
                result = InputValidator.prompt_yes_no("Test question")
                assert result is True
                # Check that error message was printed
                mock_print.assert_called()

    def test_yes_not_accepted(self):
        """Test that 'yes' is not accepted, only 'y'."""
        with patch('builtins.input', side_effect=['yes', 'y']):
            with patch('builtins.print') as mock_print:
                result = InputValidator.prompt_yes_no("Test question")
                assert result is True
                # Error message should be printed for 'yes'
                mock_print.assert_called()

    def test_no_not_accepted(self):
        """Test that 'no' is not accepted, only 'n'."""
        with patch('builtins.input', side_effect=['no', 'n']):
            with patch('builtins.print') as mock_print:
                result = InputValidator.prompt_yes_no("Test question")
                assert result is False
                # Error message should be printed for 'no'
                mock_print.assert_called()

    def test_eof_error_returns_default(self):
        """Test that EOFError returns default value."""
        with patch('builtins.input', side_effect=EOFError):
            result = InputValidator.prompt_yes_no("Test question", default=True)
            assert result is True

    def test_eof_error_returns_false_when_no_default(self):
        """Test that EOFError returns False when no default is set."""
        with patch('builtins.input', side_effect=EOFError):
            result = InputValidator.prompt_yes_no("Test question")
            assert result is False

    def test_keyboard_interrupt_returns_default(self):
        """Test that KeyboardInterrupt returns default value."""
        with patch('builtins.input', side_effect=KeyboardInterrupt):
            result = InputValidator.prompt_yes_no("Test question", default=False)
            assert result is False

    def test_keyboard_interrupt_returns_false_when_no_default(self):
        """Test that KeyboardInterrupt returns False when no default is set."""
        with patch('builtins.input', side_effect=KeyboardInterrupt):
            result = InputValidator.prompt_yes_no("Test question")
            assert result is False


class TestPromptChoice:
    """Tests for prompt_choice method."""

    def test_valid_choice(self):
        """Test that valid choice returns correct result."""
        choices = {'a': 'option_a', 'b': 'option_b', 'c': 'option_c'}
        with patch('builtins.input', return_value='b'):
            result = InputValidator.prompt_choice("Choose: ", choices)
            assert result == 'option_b'

    def test_case_insensitive_choice(self):
        """Test that choice matching is case-insensitive."""
        choices = {'a': 'option_a', 'b': 'option_b'}
        with patch('builtins.input', return_value='B'):
            result = InputValidator.prompt_choice("Choose: ", choices)
            assert result == 'option_b'

    def test_invalid_choice_then_valid(self):
        """Test that invalid choice triggers retry."""
        choices = {'a': 'option_a', 'b': 'option_b'}
        with patch('builtins.input', side_effect=['invalid', 'a']):
            with patch('builtins.print') as mock_print:
                result = InputValidator.prompt_choice("Choose: ", choices)
                assert result == 'option_a'
                mock_print.assert_called()

    def test_eof_error_returns_default(self):
        """Test that EOFError returns default value."""
        choices = {'a': 'option_a', 'b': 'option_b'}
        with patch('builtins.input', side_effect=EOFError):
            result = InputValidator.prompt_choice("Choose: ", choices, default='option_a')
            assert result == 'option_a'

    def test_eof_error_returns_empty_when_no_default(self):
        """Test that EOFError returns empty string when no default is set."""
        choices = {'a': 'option_a', 'b': 'option_b'}
        with patch('builtins.input', side_effect=EOFError):
            result = InputValidator.prompt_choice("Choose: ", choices)
            assert result == ""

    def test_keyboard_interrupt_returns_default(self):
        """Test that KeyboardInterrupt returns default value."""
        choices = {'a': 'option_a', 'b': 'option_b'}
        with patch('builtins.input', side_effect=KeyboardInterrupt):
            result = InputValidator.prompt_choice("Choose: ", choices, default='option_b')
            assert result == 'option_b'


class TestPromptWithValidation:
    """Tests for prompt_with_validation method."""

    def test_valid_input(self):
        """Test that valid input returns result from validator."""
        def validator(value):
            if value.isdigit():
                return True, int(value)
            return False, None

        with patch('builtins.input', return_value='42'):
            result = InputValidator.prompt_with_validation("Enter number: ", validator)
            assert result == 42

    def test_invalid_input_then_valid(self):
        """Test that invalid input triggers retry."""
        def validator(value):
            if value.isdigit():
                return True, int(value)
            return False, None

        with patch('builtins.input', side_effect=['abc', '123']):
            with patch('builtins.print') as mock_print:
                result = InputValidator.prompt_with_validation("Enter number: ", validator)
                assert result == 123
                mock_print.assert_called()

    def test_eof_error_returns_default(self):
        """Test that EOFError returns default value."""
        def validator(value):
            return True, value

        with patch('builtins.input', side_effect=EOFError):
            result = InputValidator.prompt_with_validation(
                "Enter: ",
                validator,
                default='default_value'
            )
            assert result == 'default_value'

    def test_keyboard_interrupt_returns_default(self):
        """Test that KeyboardInterrupt returns default value."""
        def validator(value):
            return True, value

        with patch('builtins.input', side_effect=KeyboardInterrupt):
            result = InputValidator.prompt_with_validation(
                "Enter: ",
                validator,
                default='default_value'
            )
            assert result == 'default_value'


class TestShowInvalidInputMessage:
    """Tests for _show_invalid_input_message method."""

    def test_message_printed(self):
        """Test that invalid input message is printed."""
        with patch('builtins.print') as mock_print:
            InputValidator._show_invalid_input_message()
            mock_print.assert_called_once()
            # Check that the call contains the error emoji
            call_args = mock_print.call_args[0][0]
            assert '❌' in call_args
