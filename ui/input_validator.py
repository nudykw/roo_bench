"""Input validation module for user prompts with retry logic."""

from typing import Any, Callable

from i18n import get_text


class InputValidator:
    """Validates user input with retry logic for invalid entries."""

    @staticmethod
    def prompt_yes_no(question: str, default: bool | None = None) -> bool:
        """Prompt user for yes/no confirmation with validation.

        Only accepts 'y' or 'n' as valid inputs (case-insensitive).

        Args:
            question: Question to display to user
            default: Default value if EOFError or KeyboardInterrupt occurs

        Returns:
            bool: True if user confirms with 'y', False if user confirms with 'n'
        """
        while True:
            try:
                response = input(f"{question} (y/n): ").strip().lower()
                if response == 'y':
                    return True
                if response == 'n':
                    return False
                InputValidator._show_invalid_input_message()
            except (EOFError, KeyboardInterrupt):
                return default if default is not None else False

    @staticmethod
    def prompt_choice(
        question: str,
        choices: dict[str, str],
        default: str | None = None,
    ) -> str:
        """Prompt user to choose from predefined options with validation.

        Args:
            question: Question to display to user
            choices: Dictionary mapping user input to result values
            default: Default value if EOFError or KeyboardInterrupt occurs

        Returns:
            str: The result value corresponding to user's choice
        """
        while True:
            try:
                response = input(question).strip().lower()
                if response in choices:
                    return choices[response]
                InputValidator._show_invalid_input_message()
            except (EOFError, KeyboardInterrupt):
                return default if default is not None else ""

    @staticmethod
    def prompt_with_validation(
        question: str,
        validator_func: Callable[[str], tuple[bool, Any]],
        default: Any = None,
    ) -> Any:
        """Prompt user with custom validation function.

        Args:
            question: Question to display to user
            validator_func: Function that validates input and returns (is_valid, result)
            default: Default value if EOFError or KeyboardInterrupt occurs

        Returns:
            Any: The result from validator function or default value
        """
        while True:
            try:
                response = input(question).strip()
                is_valid, result = validator_func(response)
                if is_valid:
                    return result
                InputValidator._show_invalid_input_message()
            except (EOFError, KeyboardInterrupt):
                return default

    @staticmethod
    def _show_invalid_input_message() -> None:
        """Display invalid input error message."""
        print(f"❌ {get_text('invalid_input')}")
