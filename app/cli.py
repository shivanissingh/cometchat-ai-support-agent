"""
app/cli.py — Interactive Typer CLI for the CometChat RAG Support Agent.

Commands
--------
chat
    Start an interactive support chat session.  Reads user input line-by-line,
    calls Agent.handle_message, and prints the answer.  Citations and the
    handoff recommendation are displayed after the answer.

    Options
    -------
    --debug
        After printing the answer for each turn, also print the full
        structured trace (as formatted JSON) so developers can inspect
        routing decisions, retrieval scores, conflict detection results,
        and safety-validation outcomes.

Session isolation
-----------------
A UUID is generated once at process startup and used for every turn in the
session.  This mirrors the Streamlit session model: one session per CLI
process lifetime.

Security
--------
The trace data shown by --debug is the SAME scrubbed data stored in
AgentResponse.trace — forbidden order fields and the GEMINI_API_KEY value
have already been redacted by the observability layer before they reach the
AgentResponse.  No extra scrubbing is performed here.
"""

from __future__ import annotations

import json
import uuid

import typer

app = typer.Typer(
    name="cometchat-rag-agent",
    help="CometChat RAG Support Agent CLI",
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# chat command
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

    Type your question and press Enter.  Type 'exit' or 'quit' (or press
    Ctrl-C / Ctrl-D) to end the session.
    """
    # Lazy import so that the knowledge-base indexes are only built when the
    # chat command actually runs (not on --help or other commands).
    from app.agent.orchestrator import Agent

    session_id = str(uuid.uuid4())
    typer.echo("CometChat Support Agent")
    typer.echo(f"Session: {session_id}")
    typer.echo("Type 'exit' or 'quit' to end the session.\n")

    agent = Agent()

    while True:
        try:
            user_input = typer.prompt("You")
        except (EOFError, KeyboardInterrupt):
            typer.echo("\nGoodbye!")
            break

        stripped = user_input.strip()
        if not stripped:
            continue
        if stripped.lower() in {"exit", "quit"}:
            typer.echo("Goodbye!")
            break

        response = agent.handle_message(session_id=session_id, text=stripped)

        # --- Answer -------------------------------------------------------
        typer.echo(f"\nAgent: {response.answer}")

        # --- Citations ----------------------------------------------------
        if response.citations:
            typer.echo("\nSources:")
            for cite in response.citations:
                filename = cite.get("filename", "")
                heading = cite.get("heading", "")
                typer.echo(f"  • {filename} — {heading}")

        # --- Handoff recommendation --------------------------------------
        if response.handoff:
            typer.echo(
                "\n🔺 Recommend contacting human support"
            )

        # --- Debug trace -------------------------------------------------
        if debug:
            typer.echo("\n--- Debug Trace ---")
            for event in response.trace:
                typer.echo(
                    json.dumps(
                        {
                            "stage": event.stage,
                            "turn_id": event.turn_id,
                            "timestamp": event.timestamp,
                            "payload": event.payload,
                        },
                        indent=2,
                        default=str,
                    )
                )
            typer.echo("--- End Trace ---")

        typer.echo("")  # blank line between turns


if __name__ == "__main__":
    app()
