import sys
import asyncio
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from langgraph.types import Command
from recipe_agent.utils import AsyncRateLimiter
from recipe_agent.graph.graph import graph
from recipe_agent.log_config import setup_logging

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


async def _handle_interrupt(interrupt_data, config, graph=None) -> str:
    if graph is None:
        from recipe_agent.graph.graph import graph as default_graph
        graph = default_graph

    # interrupt_data może być krotką, listą lub bezpośrednio obiektem
    if isinstance(interrupt_data, (list, tuple)):
        if len(interrupt_data) == 0:
            # brak danych — pobierz stan z checkpointera
            state = graph.get_state(config)
            parsed = state.values.get("parsed")
        else:
            payload = interrupt_data[0].value
            parsed = payload.get("parsed")
    else:
        payload = interrupt_data.value
        parsed = payload.get("parsed")

    if parsed:
        from recipe_agent.models import ParsedRecipeWithTranslations
        if isinstance(parsed, dict):
            parsed = ParsedRecipeWithTranslations(**parsed)
        from recipe_agent.graph.nodes import _build_summary
        console.print(Panel(
            _build_summary(parsed),
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
        note = Prompt.ask("Opisz co poprawić")

    resume_value = {"action": decision, "note": note}
    async for event in graph.astream(
        Command(resume=resume_value),
        config=config,
    ):
        for node_name, output in event.items():
            if node_name == "save":
                console.print(f"\n[bold green]Recipe saved![/bold green]")

    return decision


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