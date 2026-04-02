import asyncio
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from langgraph.types import Command

from recipe_agent.graph.graph import graph
from recipe_agent.logging import setup_logging

setup_logging(debug="--debug" in sys.argv)
log = structlog.get_logger()
console = Console()


async def run_recipe(raw_text: str, source_url: str | None = None) -> None:
    config = {"configurable": {"thread_id": source_url or raw_text[:40]}}

    initial_state = {
        "raw_text": raw_text,
        "source_url": source_url,
        "parsed": None,
        "parse_error": None,
        "human_decision": None,
        "human_note": None,
        "retry_count": 0,
        "messages": [],
        "saved_recipe_id": None,
    }

    async for event in graph.astream(initial_state, config=config):
        for node_name, output in event.items():
            if node_name == "__interrupt__":
                await _handle_interrupt(output, config)
                return
            log.info("graph.node.done", node=node_name)


async def _handle_interrupt(interrupt_data, config) -> None:
    payload = interrupt_data[0].value

    console.print(Panel(
        payload["summary"],
        title="[bold cyan]Podgląd przepisu[/bold cyan]",
        border_style="cyan",
    ))

    decision = Prompt.ask(
        "\nCo zrobić?",
        choices=["save", "skip", "edit"],
        default="save",
    )

    note = None
    if decision == "edit":
        note = Prompt.ask("Opisz co poprawić (LLM spróbuje jeszcze raz)")

    resume_value = {"action": decision, "note": note}
    async for event in graph.astream(
        Command(resume=resume_value),
        config=config,
    ):
        for node_name, output in event.items():
            if node_name == "save":
                console.print(f"\n[bold green]Zapisano![/bold green]")
            log.info("graph.node.done", node=node_name)


async def main() -> None:
    import sys

    if len(sys.argv) < 2:
        console.print("[red]Użycie: uv run recipe-agent <plik.txt lub URL>[/red]")
        return

    arg = sys.argv[1]

    if arg.startswith("http"):
        await run_recipe(f"Parse recipe from URL: {arg}", source_url=arg)
    else:
        with open(arg) as f:
            raw = f.read()
        await run_recipe(raw)


if __name__ == "__main__":
    asyncio.run(main())