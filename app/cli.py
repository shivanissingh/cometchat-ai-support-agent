"""
app/cli.py — Interactive CLI for the CometChat RAG Support Agent.

Commands
--------
chat
    Start an interactive support chat session. Reads user input line-by-line,
    calls Agent.handle_message, and prints the formatted answer. Citations and the
    handoff recommendation are displayed after the answer.

    Options
    -------
    --debug
        After printing the answer for each turn, also prints the full
        structured trace (as syntax-highlighted JSON) so developers can inspect
        routing decisions, retrieval scores, conflict detection results,
        and safety-validation outcomes.

Session isolation
-----------------
A UUID is generated once at process startup and used for every turn in the
session. This mirrors the Streamlit session model: one session per CLI
process lifetime.

Security
--------
The trace data shown by --debug is the SAME scrubbed data stored in
AgentResponse.trace — forbidden order fields and the GEMINI_API_KEY value
have already been redacted by the observability layer before they reach the
AgentResponse.
"""

from __future__ import annotations

import logging
import os
import warnings

# ---------------------------------------------------------------------------
# Suppress warnings, noisy logs & progress bars before importing heavy modules
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.captureWarnings(True)
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_UNAUTHENTICATED_WARNING"] = "1"

import json  # noqa: E402
import uuid  # noqa: E402

import typer  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.syntax import Syntax  # noqa: E402

# Silence root logger and attach NullHandler to avoid console log leakage
root_logger = logging.getLogger()
root_logger.setLevel(logging.CRITICAL)
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)
root_logger.addHandler(logging.NullHandler())

try:
    import huggingface_hub.utils.logging as hf_logging

    hf_logging.set_verbosity_error()
except Exception:
    pass

try:
    import transformers.utils.logging as tf_logging

    tf_logging.set_verbosity_error()
except Exception:
    pass
root_logger.setLevel(logging.CRITICAL)
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)
root_logger.addHandler(logging.NullHandler())

# Silence specific noisy third-party loggers and warnings logger
for logger_name in (
    "py.warnings",
    "app",
    "app.config",
    "app.observability",
    "app.agent",
    "observability",
    "httpx",
    "httpcore",
    "huggingface_hub",
    "sentence_transformers",
    "transformers",
    "google_genai",
    "google",
    "urllib3",
):
    log_inst = logging.getLogger(logger_name)
    log_inst.setLevel(logging.CRITICAL)
    log_inst.propagate = False

console = Console()

app = typer.Typer(
    name="cometchat-rag-agent",
    help="CometChat RAG Support Agent CLI",
    no_args_is_help=False,
)


# ---------------------------------------------------------------------------
# chat command (default entry point)
# ---------------------------------------------------------------------------


@app.command()
def chat(
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Print the full scrubbed trace after each answer.",
        is_flag=True,
    ),
) -> None:
    """Start an interactive support chat session.

    Type your question and press Enter. Type 'exit' or 'quit' (or press
    Ctrl-C / Ctrl-D) to end the session.
    """
    # Lazy import so that the knowledge-base indexes are only built when the
    # chat command actually runs
    from app.agent.orchestrator import Agent

    session_id = str(uuid.uuid4())

    mode_str = "[bold green]ENABLED[/bold green]" if debug else "[dim]DISABLED[/dim]"
    welcome_text = (
        f"[bold cyan]💬 CometChat AI Support Agent[/bold cyan]\n"
        f"[dim]Session ID:[/dim] [bright_black]{session_id}[/bright_black]\n"
        f"[dim]Debug Mode:[/dim] {mode_str}\n\n"
        f"[italic]Type your question and press Enter. Type 'exit' or 'quit' to end.[/italic]"
    )
    console.print(Panel(welcome_text, border_style="cyan", padding=(1, 2)))

    with console.status("[bold cyan]Building KB indexes...[/bold cyan]", spinner="dots"):
        agent = Agent()

    console.print("[dim green]✔ Knowledge-base ready. How can I help you today?[/dim green]\n")

    while True:
        try:
            console.print("[bold green]You[/bold green] [dim]>[/dim] ", end="")
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in {"exit", "quit"}:
            console.print("[dim]Goodbye![/dim]")
            break

        try:
            with console.status("[bold blue]Thinking...[/bold blue]", spinner="dots"):
                response = agent.handle_message(session_id=session_id, text=stripped)
        except Exception as err:
            console.print(f"\n[bold red]Error:[/bold red] {err}\n")
            continue

        # --- Answer -------------------------------------------------------
        console.print("\n[bold cyan]Agent[/bold cyan] [dim]>[/dim]")
        console.print(Markdown(response.answer))

        # --- Citations ----------------------------------------------------
        if response.citations:
            console.print("\n[bold dim]Sources:[/bold dim]")
            for cite in response.citations:
                filename = cite.get("filename", "")
                heading = cite.get("heading", "")
                console.print(f"  [cyan]•[/cyan] [bold]{filename}[/bold] [dim]— {heading}[/dim]")

        # --- Handoff recommendation --------------------------------------
        if response.handoff:
            console.print("\n[bold yellow]🔺 Recommend contacting human support[/bold yellow]")

        # --- Debug trace -------------------------------------------------
        if debug:
            turn_num = response.trace[0].turn_id if response.trace else 0
            trace_data = [
                {
                    "stage": event.stage,
                    "turn_id": event.turn_id,
                    "timestamp": event.timestamp,
                    "payload": event.payload,
                }
                for event in response.trace
            ]
            json_str = json.dumps(trace_data, indent=2, default=str)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=False)
            console.print(
                Panel(
                    syntax,
                    title=f"🔍 [bold yellow]Debug Trace — Turn {turn_num}[/bold yellow]",
                    border_style="yellow",
                    padding=(0, 1),
                )
            )

        console.print("")  # blank line between turns


if __name__ == "__main__":
    app()
