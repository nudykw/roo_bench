"""Interactive curses-based model selection interface."""

import curses
from typing import List, Any
from ui.model_selector import select_models, SelectionType


def interactive_model_select(stdscr, models: list) -> list:
    """Interactive model selection using curses with keyboard and mouse support.

    Args:
        stdscr: curses standard screen
        models: List of model dictionaries

    Returns:
        list: Selected model dictionaries
    """
    columns = [
        {'key': 'name', 'header': 'Name', 'width': 25},
        {'key': 'params', 'header': 'Params', 'width': 8},
        {'key': 'size_gb', 'header': 'Size', 'width': 10, 'formatter': lambda x: f"{x:.1f}GB" if x else "0.0GB"},
        {'key': 'max_ctx', 'header': 'Ctx', 'width': 8, 'formatter': lambda x: f"{x//1024}K" if x and x >= 1024 else str(x or "")},
        {'key': 'vision', 'header': 'Vision', 'width': 6},
        {'key': 'tools', 'header': 'Tools', 'width': 6},
    ]
    
    return select_models(
        stdscr,
        models,
        SelectionType.MULTIPLE,
        title="Select models to benchmark",
        columns=columns
    )


def select_expert_model(stdscr, models: list) -> str:
    """Select a single expert model using curses interface.

    Args:
        stdscr: curses standard screen
        models: List of model dictionaries

    Returns:
        str: Selected model name, or None to cancel.
    """
    if not models:
        return None
    
    columns = [
        {'key': 'name', 'header': 'Model', 'width': 30},
        {'key': 'params', 'header': 'Params', 'width': 10},
        {'key': 'size_gb', 'header': 'Size', 'width': 8, 'formatter': lambda x: f"{x:.1f}GB" if x else "0.0GB"},
    ]
    
    selected = select_models(
        stdscr,
        models,
        SelectionType.SINGLE,
        title="Select Expert Model",
        columns=columns
    )
    
    if not selected:
        return None
    
    return selected[0].get('name') if isinstance(selected[0], dict) else str(selected[0])
