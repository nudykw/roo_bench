"""Interactive curses-based model selection interface."""

import curses
import sys
from typing import Any, Optional

from i18n import _current_language, get_text


def interactive_model_select(stdscr: Any, models: list[dict[str, Any]], single_select: bool = False) -> list[dict[str, Any]]:
    """Interactive model selection using curses with keyboard and mouse support.

    Args:
        stdscr: curses standard screen
        models: List of model dictionaries
        single_select: If True, allow selection of only one model (used for expert evaluator)

    Returns:
        list: Selected model dictionaries
    """
    # Initialize curses features
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(True)  # Non-blocking input
    stdscr.timeout(100)  # 100ms timeout for getch

    # Enable mouse (left click only, 1 button event)
    mouse_event = 0
    try:
        mouse_event = curses.BUTTON1_CLICKED | curses.BUTTON1_DOUBLE_CLICKED
        curses.mousemask(mouse_event)
        curses.mouseinterval(0)  # No delay for double-click detection
    except Exception:
        pass

    selected: set[int] = set()  # Set of selected indices
    current_row = 0  # Currently highlighted row
    start_row = 0  # Starting row for scrolling

    # Get screen dimensions
    max_rows, max_cols = stdscr.getmaxyx()

    # Reserve lines: title (1), header (1), models (variable), hint (1), confirmation (1) = 4 minimum
    min_required = 6
    if max_rows < min_required:
        return models  # Select all if screen too small

    model_area_start = 2  # After title and header
    model_area_end = max_rows - 2  # Before hint and confirmation
    visible_models = model_area_end - model_area_start

    if visible_models <= 0:
        return models  # Select all if not enough space

    # Color pairs
    try:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Selected
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Highlighted
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Hint
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Header
        curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_WHITE)  # Selected + highlighted
    except Exception:
        pass

    def draw() -> None:
        """Draw the interface."""
        nonlocal start_row
        stdscr.erase()

        # Title
        if single_select:
            title = "Select a model for expert evaluation"
            if _current_language == "ua":
                title = "Виберіть модель для експертизи"
        else:
            title = "Select models to benchmark"
            if _current_language == "ua":
                title = "Оберіть моделі для тестування"
        try:
            stdscr.addstr(0, 0, title, curses.color_pair(4) | curses.A_BOLD)
        except curses.error:
            pass

        # Column headers
        header_fmt = "[ ] {name:<25} | {params:<5} | Size: {size_gb:4.1f} GB | MoE: {moe_str:>8} | MaxCtx: {max_ctx_str:>4} | Vision: {vision} | Tools: {tools} | Think: {thinking}"
        try:
            # Build header text
            header_model = {"name": "Name", "params": "Params", "size_gb": 0.0, "max_ctx": 32768,
                          "moe_str": "MoE", "vision": "Vision", "tools": "Tools", "thinking": "Think"}
            max_ctx_str = "32K"
            header_text = header_fmt.format(**header_model, max_ctx_str=max_ctx_str)
            # Truncate to screen width
            header_text = header_text[:max_cols-1]
            stdscr.addstr(1, 0, header_text, curses.color_pair(4) | curses.A_UNDERLINE)
        except (curses.error, ValueError, KeyError):
            pass

        # Calculate visible range
        if current_row < start_row:
            start_row = current_row
        elif current_row >= start_row + visible_models:
            start_row = current_row - visible_models + 1

        # Draw models
        for i in range(visible_models):
            model_idx = start_row + i
            if model_idx >= len(models):
                break

            y = model_area_start + i
            m = models[model_idx]
            
            # Ensure size_gb is a number (handle cases where it might be a string)
            try:
                size_gb = float(m.get('size_gb', 0))
            except (ValueError, TypeError):
                size_gb = 0.0
            
            max_ctx = m.get('max_ctx', 32768)
            if isinstance(max_ctx, str):
                try:
                    max_ctx = int(max_ctx)
                except (ValueError, TypeError):
                    max_ctx = 32768
            max_ctx_str = f"{max_ctx // 1024}K" if max_ctx >= 1024 else str(max_ctx)

            is_selected = model_idx in selected
            is_current = model_idx == current_row
            
            # Get MoE data and format it
            moe_data = m.get('moe', None)
            if moe_data is None:
                moe_str = "❓"
            elif moe_data is False:
                moe_str = "⬛"
            elif isinstance(moe_data, dict):
                num_experts = moe_data.get('num_experts', 'N/A')
                if isinstance(num_experts, int) and num_experts > 0:
                    moe_str = f"🟡({num_experts})"
                else:
                    moe_str = "🟡"
            elif moe_data is True:
                moe_str = "🟡"
            else:
                moe_str = "❓"

            # Format line with index
            select_marker = "[x]" if is_selected else "[ ]"
            line_fmt = f"{select_marker} {{name:<25}} | {{params:<5}} | Size: {{size_gb:4.1f}} GB | MoE: {{moe_str:>8}} | MaxCtx: {{max_ctx_str:>4}} | Vision: {{vision}} | Tools: {{tools}} | Think: {{thinking}}"

            try:
                # Create a clean dict without size_gb to avoid string/float conflict
                fmt_data = {k: v for k, v in m.items() if k != 'size_gb'}
                fmt_data['size_gb'] = size_gb
                fmt_data['max_ctx_str'] = max_ctx_str
                fmt_data['moe_str'] = moe_str
                line_text = line_fmt.format(**fmt_data)
                line_text = line_text[:max_cols-1]

                if is_selected and is_current:
                    attr = curses.color_pair(5) | curses.A_BOLD
                elif is_current:
                    attr = curses.color_pair(2)
                elif is_selected:
                    attr = curses.color_pair(1)
                else:
                    attr = curses.A_NORMAL

                stdscr.addstr(y, 0, line_text.ljust(max_cols-1), attr)
            except (curses.error, ValueError, KeyError) as e:
                # Log format errors for debugging
                print(f"⚠️  Format error for model {m.get('name', 'unknown')}: {e}", file=sys.stderr)
                pass

        # Draw hint
        hint_key = "select_models_interactive"
        hint = get_text(hint_key)
        hint = hint[:max_cols-1]
        try:
            stdscr.addstr(max_rows - 2, 0, hint.ljust(max_cols-1), curses.color_pair(3))
        except curses.error:
            pass

        # Confirmation line
        selected_count = len(selected)
        if single_select:
            if selected_count > 0:
                confirm = "Selected: 1 model. Press Enter to confirm, Esc to cancel"
            else:
                confirm = "No model selected. Press Enter to select first, Esc to cancel"
        else:
            if selected_count > 0:
                confirm = f"Selected: {selected_count} models. Press Enter to confirm, Esc to deselect all"
            else:
                confirm = "No models selected. Press Enter to select all, Esc to cancel"
        confirm = confirm[:max_cols-1]
        try:
            stdscr.addstr(max_rows - 1, 0, confirm.ljust(max_cols-1), curses.A_BOLD)
        except curses.error:
            pass

        stdscr.refresh()

    # Main loop
    while True:
        draw()

        key = stdscr.getch()

        # Handle mouse events
        if key == -1:  # No keyboard input, check mouse
            try:
                bstate, x, y, z, btype = curses.getmouse()
                if btype & mouse_event:
                    # Check if click is on a model line
                    if model_area_start <= y < model_area_end:
                        model_idx = start_row + (y - model_area_start)
                        if 0 <= model_idx < len(models):
                            if single_select:
                                selected.clear()
                                selected.add(model_idx)
                                return [models[model_idx]]
                            else:
                                if model_idx in selected:
                                    selected.discard(model_idx)
                                else:
                                    selected.add(model_idx)
                            current_row = model_idx
                            if not single_select:
                                continue
                            else:
                                continue
            except curses.error:
                pass
            except KeyboardInterrupt:
                break
        elif key == ord('\n') or key == ord('\r'):  # Enter - confirm
            if single_select:
                if selected:
                    return [models[i] for i in sorted(selected)]
                else:
                    return [models[current_row]]
            else:
                if selected:
                    return [models[i] for i in sorted(selected)]
                else:
                    return models
        elif key == 27:  # Esc - deselect all
            selected.clear()
            current_row = 0
            start_row = 0
        elif key == ord('a') or key == ord('A'):  # 'a' - select all (only in multi-select mode)
            if not single_select:
                selected = set(range(len(models)))
        elif key == curses.KEY_UP:  # Up arrow
            if current_row > 0:
                current_row -= 1
        elif key == curses.KEY_DOWN:  # Down arrow
            if current_row < len(models) - 1:
                current_row += 1
        elif key == curses.KEY_PPAGE:  # Page Up
            current_row = max(0, current_row - visible_models)
        elif key == curses.KEY_NPAGE:  # Page Down
            current_row = min(len(models) - 1, current_row + visible_models)
        elif key == ord(' '):  # Space - toggle selection
            if single_select:
                selected.clear()
                selected.add(current_row)
            else:
                if current_row in selected:
                    selected.discard(current_row)
                else:
                    selected.add(current_row)


def select_expert_model(stdscr: Any, models: list[dict[str, Any]]) -> str | None:
    """Select a single expert model using curses interface.

    Args:
        stdscr: curses standard screen
        models: List of model dictionaries

    Returns:
        str | None: Selected model name, or None to cancel.
    """
    if not models:
        return None

    selected = interactive_model_select(stdscr, models, single_select=True)

    if not selected:
        return None

    return selected[0]['name']  # type: ignore[no-any-return]


def select_retest_decision(stdscr: Any, model_name: str, tested_count: int, total_count: int) -> str:
    """Select retest decision using curses interface.

    Args:
        stdscr: curses standard screen
        model_name: Name of the model being checked
        tested_count: Number of models already tested
        total_count: Total number of models to test

    Returns:
        str: Selected decision value ('yes', 'no', 'yes_all', 'no_all')
    """
    from export.retest_dialog import RetestDecision
    
    options = [
        (RetestDecision.YES, get_text('retest_yes')),
        (RetestDecision.NO, get_text('retest_no')),
        (RetestDecision.YES_ALL, get_text('retest_yes_all')),
        (RetestDecision.NO_ALL, get_text('retest_no_all')),
    ]
    
    # Initialize curses features
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)
    
    # Enable mouse
    try:
        mouse_event = curses.BUTTON1_CLICKED | curses.BUTTON1_DOUBLE_CLICKED
        curses.mousemask(mouse_event)
        curses.mouseinterval(0)
    except Exception:
        mouse_event = 0
    
    current_idx = 0
    
    # Get screen dimensions
    max_rows, max_cols = stdscr.getmaxyx()
    
    # Color pairs
    try:
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)
    except Exception:
        pass
    
    def draw() -> None:
        stdscr.erase()
        
        # Header
        header = f"⚠️  Model '{model_name}' already exists in results file!"
        subheader = f"   Progress: {tested_count}/{total_count} models tested"
        try:
            stdscr.addstr(0, 0, header[:max_cols-1], curses.color_pair(4) | curses.A_BOLD)
            stdscr.addstr(1, 0, subheader[:max_cols-1], curses.A_NORMAL)
        except curses.error:
            pass
        
        # Prompt
        prompt = get_text('retest_prompt', model=model_name)
        try:
            stdscr.addstr(3, 0, prompt[:max_cols-1])
        except curses.error:
            pass
        
        # Options
        hint = "Use ↑/↓ arrows to navigate, Enter to select"
        if _current_language == "ua":
            hint = "Використовуйте ↑/↓ для навігації, Enter для вибору"
        
        try:
            stdscr.addstr(max_rows - 2, 0, hint[:max_cols-1], curses.color_pair(3))
        except curses.error:
            pass
        
        for i, (decision, label) in enumerate(options):
            y = 5 + i
            if y >= max_rows - 3:
                break
            try:
                if i == current_idx:
                    stdscr.addstr(y, 2, f"▶ {label}", curses.color_pair(2) | curses.A_BOLD)
                else:
                    stdscr.addstr(y, 2, f"  {label}", curses.A_NORMAL)
            except curses.error:
                pass
        
        stdscr.refresh()
    
    while True:
        draw()
        
        key = stdscr.getch()
        
        if key == curses.KEY_UP:
            current_idx = (current_idx - 1) % len(options)
        elif key == curses.KEY_DOWN:
            current_idx = (current_idx + 1) % len(options)
        elif key == ord('\n') or key == ord('\r'):
            return options[current_idx][0].value
        elif key == 27:  # Esc
            return RetestDecision.NO.value
