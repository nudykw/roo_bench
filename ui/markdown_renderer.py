"""Markdown rendering module for AI analysis output."""

from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from typing import Iterator

console = Console()


def display_markdown(text: str) -> None:
    """Display formatted Markdown text to console.
    
    Args:
        text: Markdown-formatted text to display
    """
    console.print()
    console.print(Markdown(text))
    console.print()


def stream_markdown(chunks: Iterator[str]) -> str:
    """Stream markdown chunks and display progress, return accumulated text.
    
    Args:
        chunks: Iterator of markdown text chunks
        
    Returns:
        str: Accumulated markdown text
    """
    accumulated = []
    total_tokens = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Генерация...", total=100)
        
        for chunk in chunks:
            accumulated.append(chunk)
            total_tokens += 1
            # Update progress with token count
            progress.update(task, description=f"Генерация... {total_tokens} токенов", advance=1)
    
    # Display final markdown
    display_markdown("".join(accumulated))
    return "".join(accumulated)
