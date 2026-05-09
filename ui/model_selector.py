"""Custom model selection interface with configurable selection modes."""

import curses
import sys
from enum import Enum
from typing import List, Dict, Optional, Any
from i18n import get_text, _current_language


class SelectionType(Enum):
    """Type of selection."""
    SINGLE = "single"
    MULTIPLE = "multiple"


def get_item_value(item: Any, key: str) -> Any:
    """Extract value from item by key.
    
    Supports both dict and object types.
    """
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def wrap_text(text: str, width: int) -> List[str]:
    """Wrap text to fit within column width.
    
    Splits text into lines, breaking at word boundaries when possible.
    """
    if not text:
        return ['']
    
    text = str(text)
    if len(text) <= width:
        return [text]
    
    lines = []
    words = text.split(' ')
    current_line = ""
    
    for word in words:
        if len(current_line + word) <= width:
            current_line += word + ' '
        else:
            if current_line:
                lines.append(current_line.strip())
            current_line = word + ' '
    
    if current_line:
        lines.append(current_line.strip())
    
    return lines if lines else ['']


def format_line_with_wrapping(item: Any, columns: List[Dict]) -> List[str]:
    """Format a line with text wrapping support.
    
    Returns list of visual lines (one item may span multiple lines).
    """
    column_lines: List[List[str]] = []
    max_lines = 0
    
    for col in columns:
        key = col['key']
        width = col.get('width', 20)
        formatter = col.get('formatter', str)
        
        value = get_item_value(item, key)
        formatted = formatter(value) if callable(formatter) and value is not None else str(value or '')
        
        wrapped = wrap_text(formatted, width)
        column_lines.append(wrapped)
        max_lines = max(max_lines, len(wrapped))
    
    # Pad each column to same height
    for col_lines in column_lines:
        while len(col_lines) < max_lines:
            col_lines.append(' ' * columns[len(col_lines)].get('width', 20))
    
    # Combine columns into visual lines
    result = []
    for i in range(max_lines):
        parts = [col_lines[i] for col_lines in column_lines]
        result.append(' | '.join(parts))
    
    return result


def select_models(
    stdscr,
    items: List[Any],
    selection_type: SelectionType,
    title: str,
    columns: List[Dict],
    preselected: Optional[List[str]] = None,
    value_key: str = 'name'
) -> List[Any]:
    """Custom selection interface with configurable selection modes.
    
    Args:
        stdscr: curses standard screen
        items: List of items to select from (any type)
        selection_type: SelectionType.SINGLE or SelectionType.MULTIPLE
        title: Title for the selection dialog
        columns: List of column configurations
        preselected: Optional list of preselected values
        value_key: Key to use for value identification (default: 'name')
        
    Returns:
        list: List of selected items (references to original items)
    """
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)
    
    # Enable mouse
    try:
        mouse_event = curses.BUTTON1_CLICKED | curses.BUTTON1_DOUBLE_CLICKED
        curses.mousemask(mouse_event)
        curses.mouseinterval(0)
    except Exception:
        pass
    
    # Initialize colors
    try:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_WHITE)
    except Exception:
        pass
    
    # Handle empty list
    if not items:
        return []
    
    is_single = selection_type == SelectionType.SINGLE
    
    # Initialize selection
    selected = set()
    if preselected:
        preselected_set = set(preselected)
        for i, item in enumerate(items):
            item_value = get_item_value(item, value_key)
            if item_value in preselected_set:
                selected.add(i)
    
    current_row = 0
    start_row = 0
    
    # Get screen dimensions
    max_rows, max_cols = stdscr.getmaxyx()
    
    min_required = 6
    if max_rows < min_required:
        return items
    
    model_area_start = 2
    model_area_end = max_rows - 2
    visible_models = model_area_end - model_area_start
    
    if visible_models <= 0:
        return items
    
    def draw():
        nonlocal start_row
        stdscr.erase()
        
        # Title
        try:
            stdscr.addstr(0, 0, title, curses.color_pair(4) | curses.A_BOLD)
        except curses.error:
            pass
        
        # Column headers
        try:
            header_parts = []
            for col in columns:
                header_parts.append(col.get('header', col['key']).ljust(col.get('width', 20)))
            header_text = ' | '.join(header_parts)[:max_cols - 1]
            stdscr.addstr(1, 0, header_text, curses.color_pair(4) | curses.A_UNDERLINE)
        except (curses.error, KeyError):
            pass
        
        # Scroll adjustment
        if current_row < start_row:
            start_row = current_row
        elif current_row >= start_row + visible_models:
            start_row = current_row - visible_models + 1
        
        # Draw models
        for i in range(visible_models):
            model_idx = start_row + i
            if model_idx >= len(items):
                break
            
            y = model_area_start + i
            item = items[model_idx]
            
            is_selected = model_idx in selected
            is_current = model_idx == current_row
            
            # Get selection marker
            if is_single:
                marker = "[●]" if is_selected else "[○]"
            else:
                marker = "[x]" if is_selected else "[ ]"
            
            # Format line
            try:
                lines = format_line_with_wrapping(item, columns)
                for line_idx, line in enumerate(lines):
                    line_y = y + line_idx
                    if line_y >= model_area_end:
                        break
                    
                    # Add marker to first line
                    if line_idx == 0:
                        display_line = marker + ' ' + line
                    else:
                        display_line = ' ' * 4 + line
                    
                    display_line = display_line[:max_cols - 1]
                    
                    if is_selected and is_current:
                        attr = curses.color_pair(5) | curses.A_BOLD
                    elif is_current:
                        attr = curses.color_pair(2)
                    elif is_selected:
                        attr = curses.color_pair(1)
                    else:
                        attr = curses.A_NORMAL
                    
                    stdscr.addstr(line_y, 0, display_line.ljust(max_cols - 1), attr)
            except (curses.error, KeyError, ValueError):
                pass
        
        # Hint
        try:
            if is_single:
                hint = get_text("select_model_single")
            else:
                hint = get_text("select_models_interactive")
            stdscr.addstr(max_rows - 2, 0, hint[:max_cols - 1], curses.color_pair(3))
        except curses.error:
            pass
        
        # Confirmation
        try:
            selected_count = len(selected)
            if is_single:
                if selected_count > 0:
                    confirm = f"Selected: {items[list(selected)[0]].get('name', 'item')}"
                else:
                    confirm = "No model selected"
            else:
                if selected_count > 0:
                    confirm = f"Selected: {selected_count} models. Enter: confirm, Esc: cancel"
                else:
                    confirm = "No models selected. Enter: select all, Esc: cancel"
            stdscr.addstr(max_rows - 1, 0, confirm[:max_cols - 1], curses.A_BOLD)
        except curses.error:
            pass
        
        stdscr.refresh()
    
    # Main loop
    while True:
        draw()
        
        key = stdscr.getch()
        
        # Mouse events
        if key == -1:
            try:
                bstate, x, y, z, btype = curses.getmouse()
                if btype & (curses.BUTTON1_CLICKED | curses.BUTTON1_DOUBLE_CLICKED):
                    if model_area_start <= y < model_area_end:
                        model_idx = start_row + (y - model_area_start)
                        if 0 <= model_idx < len(items):
                            if is_single:
                                selected.clear()
                                selected.add(model_idx)
                            else:
                                if model_idx in selected:
                                    selected.discard(model_idx)
                                else:
                                    selected.add(model_idx)
                            current_row = model_idx
                            continue
            except curses.error:
                pass
            except KeyboardInterrupt:
                break
        elif key == ord('\n') or key == ord('\r'):
            if is_single:
                if selected:
                    return [items[i] for i in selected]
                return []
            else:
                if selected:
                    return [items[i] for i in sorted(selected)]
                return items
        elif key == 27:
            selected.clear()
            current_row = 0
            start_row = 0
        elif key == curses.KEY_UP:
            if current_row > 0:
                current_row -= 1
        elif key == curses.KEY_DOWN:
            if current_row < len(items) - 1:
                current_row += 1
        elif key == curses.KEY_PPAGE:
            current_row = max(0, current_row - visible_models)
        elif key == curses.KEY_NPAGE:
            current_row = min(len(items) - 1, current_row + visible_models)
        elif key == ord(' '):
            if is_single:
                selected.clear()
                selected.add(current_row)
            else:
                if current_row in selected:
                    selected.discard(current_row)
                else:
                    selected.add(current_row)