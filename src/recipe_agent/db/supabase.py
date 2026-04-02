import structlog
from supabase import create_client, Client
from recipe_agent.config import settings
from recipe_agent.models import ParsedRecipeWithTranslations, Language

log = structlog.get_logger()


_sb_client = None


def get_client() -> Client:
    global _sb_client
    if _sb_client is None:
        _sb_client = create_client(settings.supabase_url, settings.supabase_service_key)
    return _sb_client


async def upsert_ingredient(sb: Client, name: str, default_unit: str | None, name_i18n: dict) -> str:
    result = sb.table("ingredients").upsert(
        {"name": name, "default_unit": default_unit, "name_i18n": name_i18n},
        on_conflict="name",
    ).execute()
    return result.data[0]["id"]


async def upsert_item(sb: Client, name: str, tag: str, name_i18n: dict) -> str:
    result = sb.table("items").upsert(
        {"name": name, "tag": tag, "name_i18n": name_i18n},
        on_conflict="name",
    ).execute()
    return result.data[0]["id"]


async def save_full_recipe(sb: Client, parsed: ParsedRecipeWithTranslations) -> str:
    log.info("db.save.start", title=parsed.title)

    recipe_row = sb.table("recipes").insert({
        "title": parsed.title,
        "title_i18n": {Language.PL: parsed.title_i18n.get(Language.PL)},
        "description": parsed.description,
        "description_i18n": parsed.description_i18n,
        "difficulty_level": parsed.difficulty_level,
        "duration_minutes": parsed.duration_minutes,
        "category": parsed.category,
        "area": parsed.area,
        "tags": parsed.tags,
        "source_url": parsed.source_url,
        "image_url": parsed.image_url,
        "youtube_url": parsed.youtube_url,
    }).execute()
    recipe_id = recipe_row.data[0]["id"]

    for sort_order, ing in enumerate(parsed.ingredients):
        ing_id = await upsert_ingredient(
            sb, ing.name, ing.unit,
            name_i18n={},
        )
        sb.table("recipe_ingredients").insert({
            "recipe_id": recipe_id,
            "ingredient_id": ing_id,
            "amount": ing.amount,
            "unit": ing.unit,
            "sort_order": sort_order,
        }).execute()

    for step in parsed.steps:
        step_row = sb.table("recipe_steps").insert({
            "recipe_id": recipe_id,
            "step_number": step.step_number,
            "instruction": step.instruction,
            "instruction_i18n": step.instruction_i18n,
            "duration_seconds": step.duration_seconds,
        }).execute()
        step_id = step_row.data[0]["id"]

        for ing in step.ingredients:
            ing_id = await upsert_ingredient(
                sb, ing.name, ing.unit,
                name_i18n=ing.name_i18n,
            )
            sb.table("step_ingredients").insert({
                "step_id": step_id,
                "ingredient_id": ing_id,
                "amount": ing.amount,
                "unit": ing.unit,
                "actions": [a.value for a in ing.actions],
            }).execute()

        for item in step.items:
            item_id = await upsert_item(
                sb, item.name, item.tag,
                name_i18n=item.name_i18n,
            )
            sb.table("step_items").insert({
                "step_id": step_id,
                "item_id": item_id,
            }).execute()

    log.info("db.save.ok", recipe_id=recipe_id, title=parsed.title)
    return recipe_id


async def update_recipe_i18n(
    sb: Client,
    recipe_id: str,
    parsed: ParsedRecipeWithTranslations,
) -> None:
    sb.table("recipes").update({
        "title_i18n": parsed.title_i18n,
        "description_i18n": parsed.description_i18n,
    }).eq("id", recipe_id).execute()
    log.info("db.update_recipe_i18n.ok", recipe_id=recipe_id)


async def save_steps(
    sb: Client,
    recipe_id: str,
    parsed: ParsedRecipeWithTranslations,
) -> None:
    log.info("db.save_steps.start", recipe_id=recipe_id, steps=len(parsed.steps))

    try:
        for step in parsed.steps:
            step_row = sb.table("recipe_steps").insert({
                "recipe_id": recipe_id,
                "step_number": step.step_number,
                "instruction": step.instruction,
                "instruction_i18n": step.instruction_i18n,
                "duration_seconds": step.duration_seconds,
            }).execute()
            step_id = step_row.data[0]["id"]

            for ing in step.ingredients:
                ing_id = await upsert_ingredient(sb, ing.name, ing.unit, ing.name_i18n)
                sb.table("step_ingredients").insert({
                    "step_id": step_id,
                    "ingredient_id": ing_id,
                    "amount": ing.amount,
                    "unit": ing.unit,
                    "actions": [a.value for a in ing.actions],
                }).execute()

            for item in step.items:
                item_id = await upsert_item(sb, item.name, item.tag, item.name_i18n)
                sb.table("step_items").insert({
                    "step_id": step_id,
                    "item_id": item_id,
                }).execute()
    except Exception as e:
        log.error("db.save_steps.error", error=str(e), recipe_id=recipe_id)
        sb.table("recipe_steps").delete().eq("recipe_id", recipe_id).execute()
        raise

    log.info("db.save_steps.ok", recipe_id=recipe_id)