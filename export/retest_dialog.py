"""Dialog for retest decision when model already exists in results."""

from enum import Enum
from i18n import get_text


class RetestDecision(Enum):
    """Possible decisions for retesting an existing model."""
    YES = "yes"          # Retest this model
    NO = "no"            # Skip this model
    YES_ALL = "yes_all"  # Retest all remaining models
    NO_ALL = "no_all"    # Skip all remaining models


def prompt_retest_decision(model_name: str, tested_count: int, total_count: int) -> RetestDecision:
    """Prompt user for retest decision when model already exists in results.

    Args:
        model_name: Name of the model being checked.
        tested_count: Number of models already tested.
        total_count: Total number of models to test.

    Returns:
        RetestDecision enum value indicating user's choice.
    """
    print(f"\n{'='*60}")
    print(f"⚠️  Model '{model_name}' already exists in results file!")
    print(f"   Progress: {tested_count}/{total_count} models tested")
    print(f"{'='*60}")
    
    print(f"\n{get_text('retest_prompt', model=model_name)}")
    print(f"  1. {get_text('retest_yes')}")
    print(f"  2. {get_text('retest_no')}")
    print(f"  3. {get_text('retest_yes_all')}")
    print(f"  4. {get_text('retest_no_all')}")
    
    while True:
        try:
            choice = input(f"\n{get_text('retest_select')} ").strip().lower()
            
            if choice in ('1', 'yes', 'y'):
                return RetestDecision.YES
            elif choice in ('2', 'no', 'n'):
                return RetestDecision.NO
            elif choice in ('3', 'yes_all', 'ya'):
                return RetestDecision.YES_ALL
            elif choice in ('4', 'no_all', 'na'):
                return RetestDecision.NO_ALL
            else:
                print(f"❌ {get_text('invalid_input')}")
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