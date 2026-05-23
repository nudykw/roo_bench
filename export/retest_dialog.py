"""Dialog for retest decision when model already exists in results."""

import curses
from enum import Enum

from i18n import _current_language, get_text
from ui.input_validator import InputValidator


class RetestDecision(Enum):
    """Possible decisions for retesting an existing model."""
    YES = "yes"          # Retest this model
    NO = "no"            # Skip this model
    YES_ALL = "yes_all"  # Retest all remaining models
    NO_ALL = "no_all"    # Skip all remaining models


def prompt_retest_decision(model_name: str, tested_count: int, total_count: int) -> RetestDecision:
    """Prompt user for retest decision when model already exists in results.
    Uses curses-based single-selection menu with keyboard navigation.

    Args:
        model_name: Name of the model being checked.
        tested_count: Number of models already tested.
        total_count: Total number of models to test.

    Returns:
        RetestDecision enum value indicating user's choice.
    """
    from ui.curses_selector import select_retest_decision
    
    try:
        decision_value = curses.wrapper(
            lambda stdscr: select_retest_decision(stdscr, model_name, tested_count, total_count)
        )
        return RetestDecision(decision_value)
    except curses.error:
        # Fallback to simple text input if curses fails
        return _prompt_retest_decision_fallback(model_name, tested_count, total_count)


def _prompt_retest_decision_fallback(model_name: str, tested_count: int, total_count: int) -> RetestDecision:
    """Fallback text-based prompt when curses is not available."""
    options = [
        (RetestDecision.YES, get_text('retest_yes')),
        (RetestDecision.NO, get_text('retest_no')),
        (RetestDecision.YES_ALL, get_text('retest_yes_all')),
        (RetestDecision.NO_ALL, get_text('retest_no_all')),
    ]
    
    current_idx = 0
    
    # Print header
    print(f"\n{'='*60}")
    print(f"⚠️  Model '{model_name}' already exists in results file!")
    print(f"   Progress: {tested_count}/{total_count} models tested")
    print(f"{'='*60}")
    
    print(f"\n{get_text('retest_prompt', model=model_name)}")
    
    hint = "Use ↑/↓ arrows to navigate, Enter to select"
    if _current_language == "ua":
        hint = "Використовуйте ↑/↓ для навігації, Enter для вибору"
    
    while True:
        # Draw options
        for i, (decision, label) in enumerate(options):
            if i == current_idx:
                print(f"  ▶ {label}")
            else:
                print(f"    {label}")
        
        print(f"\n  {hint}")
        
        try:
            choice = input(f"\n  {get_text('retest_select')} ").strip().lower()
            
            if choice in ('1', 'k', 'up'):
                current_idx = (current_idx - 1) % 4
            elif choice in ('2', 'j', 'down'):
                current_idx = (current_idx + 1) % 4
            elif choice in ('', 'enter', 'y', 'yes'):
                return options[current_idx][0]
            elif choice in ('n', 'no'):
                return RetestDecision.NO
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < 4:
                        return options[idx][0]
                except ValueError:
                    pass
                InputValidator._show_invalid_input_message()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{get_text('save_cancelled')}")
            return RetestDecision.NO


def should_skip_model(decision: RetestDecision, current_index: int, total_count: int) -> bool:
    """Determine if model should be skipped based on user decision.

    Args:
        decision: User's retest decision.
        current_index: Current model index (0-based).
        total_count: Total number of models.

    Returns:
        True if model should be skipped, False otherwise.
    """
    if decision == RetestDecision.NO:
        return True
    elif decision == RetestDecision.NO_ALL:
        return True
    elif decision == RetestDecision.YES_ALL:
        return False
    elif decision == RetestDecision.YES:
        return False
    return True


def should_stop_testing(decision: RetestDecision) -> bool:
    """Check if testing should stop (NO_ALL decision).

    Args:
        decision: User's retest decision.

    Returns:
        True if testing should stop, False otherwise.
    """
    return decision == RetestDecision.NO_ALL