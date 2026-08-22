"""
app/cli.py — Placeholder Typer CLI application.

Later agents will add commands for ingestion, retrieval testing,
evaluation runs, and interactive chat sessions.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="cometchat-rag-agent",
    help="CometChat RAG Support Agent CLI",
    no_args_is_help=True,
)


@app.command()
def hello() -> None:
    """Placeholder command — confirms the CLI entry point is reachable."""
    typer.echo("CometChat RAG Support Agent — CLI placeholder. Build in progress.")


if __name__ == "__main__":
    app()
