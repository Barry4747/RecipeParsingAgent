import asyncio
import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from langgraph.types import Command

from recipe_agent.config import settings
from recipe_agent.db.supabase import get_client
from recipe_agent.graph.graph import migration_graph
from recipe_agent.logging import setup_logging
from recipe_agent.main import _handle_interrupt

log = structlog.get_logger()
console = Console()


async def fetch_pending(sb) -> list[dict]:
    all_old = sb.table("recipes_old").select(
        "id, title, recipe_plaintext, description, category, area, duration_minutes, difficulty_level"
    ).limit(100000).execute().data

    done_ids = {
        str(r["recipe_id"])
        for r in sb.table("recipe_steps").select("recipe_id").limit(100000).execute().data
    }

    pending = [r for r in all_old if str(r["id"]) not in done_ids]
    log.info("migrate.pending", total=len(all_old), pending=len(pending), done=len(done_ids))
    return pending


def build_raw_text(recipe: dict) -> str:
    lines = [f"Title: {recipe['title']}"]
    if recipe.get("description"):
        lines.append(f"Description: {recipe['description']}")
    if recipe.get("category"):
        lines.append(f"Category: {recipe['category']}")
    if recipe.get("area"):
        lines.append(f"Cuisine: {recipe['area']}")
    if recipe.get("duration_minutes"):
        lines.append(f"Duration: {recipe['duration_minutes']} minutes")
    if recipe.get("difficulty_level"):
        lines.append(f"Difficulty: {recipe['difficulty_level']}")
    if recipe.get("recipe_plaintext"):
        lines.append(f"\nInstructions:\n{recipe['recipe_plaintext']}")
    return "\n".join(lines)


async def run_migration(batch_size: int = 0, auto_save: bool = False) -> None:
    setup_logging()
    sb = get_client()

    pending = await fetch_pending(sb)
    if not pending:
        console.print("[green]Wszystkie przepisy mają już kroki![/green]")
        return

    to_process = pending[:batch_size] if batch_size > 0 else pending
    console.print(f"\n[cyan]Do przetworzenia: {len(to_process)} przepisów[/cyan]\n")

    ok = skipped = failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Migracja...", total=len(to_process))

        for recipe in to_process:
            structlog.contextvars.bind_contextvars(recipe_title=recipe["title"])
            progress.update(task, description=f"[cyan]{recipe['title'][:50]}[/cyan]")

            config = {"configurable": {"thread_id": recipe["id"]}}
            initial_state = {
                "raw_text": build_raw_text(recipe),
                "source_url": None,
                "recipe_id": recipe["id"],
                "parsed": None,
                "parse_error": None,
                "human_decision": None,
                "human_note": None,
                "retry_count": 0,
                "messages": [],
                "saved_recipe_id": None,
            }

            try:
                async for event in migration_graph.astream(initial_state, config=config):
                    for node_name, output in event.items():
                        if node_name == "__interrupt__":
                            if auto_save:
                                async for _ in migration_graph.astream(
                                    Command(resume={"action": "save", "note": None}),
                                    config=config,
                                ):
                                    pass
                                ok += 1
                            else:
                                progress.stop()
                                decision = await _handle_interrupt(output, config, migration_graph)
                                progress.start()
                                if decision == "save":
                                    ok += 1
                                else:
                                    skipped += 1

            except Exception as e:
                log.error("migrate.recipe.failed", error=str(e))
                failed += 1

            structlog.contextvars.clear_contextvars()
            progress.advance(task)

    console.print(f"\n[bold]Wynik migracji:[/bold]")
    console.print(f"   Zapisano:  {ok}")
    console.print(f"   Pominięto: {skipped}")
    console.print(f"   Błędy:     {failed}")


if __name__ == "__main__":
    import sys
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    auto = "--auto" in sys.argv
    asyncio.run(run_migration(batch_size=batch, auto_save=auto))