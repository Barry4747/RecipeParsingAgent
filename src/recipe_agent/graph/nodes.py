import structlog
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from recipe_agent.config import settings
from recipe_agent.models import ParsedRecipe
from recipe_agent.graph.state import AgentState

log = structlog.get_logger()

_llm = ChatOllama(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
    temperature=0,
    num_predict=4096,
    num_ctx=8192,
).with_structured_output(ParsedRecipe, include_raw=True)

_PARSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a culinary recipe parser. Extract structured data from recipe text.
Split the recipe into short steps, each step should be a single action.
Rules for steps:
- step.ingredients = ingredients involved in this step. For each ingredient, you must fill the `actions` field. This is an ordered list of actions performed on the ingredient in this specific step.
  Available actions:
    * 'peel': peeling skin off (e.g., potatoes, apples).
    * 'slice': cutting into slices or discs (e.g., onions, tomatoes).
    * 'chop': cutting into chunks or pieces (e.g., vegetables, meat).
    * 'mince': very fine chopping (e.g., garlic, herbs).
    * 'grate': shredding using a grater (e.g., cheese, carrots).
    * 'blend': processing until smooth (e.g., soups, smoothies).
    * 'melt': melting a solid ingredient (e.g., butter, chocolate).
    * 'add': physically adding the ingredient to the dish, pot, pan or bowl. IMPORTANT: If the ingredient is only being prepared (e.g., chopped) but NOT added to the main dish in this step, do NOT include 'add'.
- step.items = only tools ACTIVELY USED (max 1 per tag per step)
- "mix", "wait", "rest" steps usually have NO ingredients unless being added
- Normalize ingredient names to lowercase

Return valid JSON matching the schema exactly."""),
    ("human", "{input}"),
])


@retry(
    retry=retry_if_exception_type((ValueError, KeyError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
async def _parse_with_retry(text: str) -> ParsedRecipe:
    chain = _PARSE_PROMPT | _llm
    result = await chain.ainvoke({"input": text})

    if result["parsing_error"]:
        raise ValueError(f"Schema mismatch: {result['parsing_error']}")

    parsed = result["parsed"]
    if parsed is None:
        raise ValueError("Model zwrócił None zamiast sparsowanego przepisu")

    return parsed


async def node_parse(state: AgentState) -> dict:
    log.info("node.parse.start", source_url=state.get("source_url"))

    input_text = state["raw_text"]
    if state.get("human_note"):
        input_text += f"\n\nCorrection from user: {state['human_note']}"

    try:
        parsed = await _parse_with_retry(input_text)
        log.info("node.parse.ok", title=parsed.title, steps=len(parsed.steps))
        return {
            "parsed": parsed,
            "parse_error": None,
            "retry_count": state.get("retry_count", 0) + 1,
        }
    except Exception as e:
        log.error("node.parse.failed", error=str(e))
        return {
            "parsed": None,
            "parse_error": str(e),
        }


from recipe_agent.models import ParsedRecipeWithTranslations, Language, RecipeStep

_TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a culinary translator specializing in Polish.
Translate the given JSON fields from English to Polish.

Rules:
- Keep culinary terms accurate (e.g. "fold" → "delikatnie wmieszaj", "sauté" → "podsmaż")
- Ingredient names should be in nominative case (mianownik): "onion" → "cebula"
- Tool names should be in nominative case: "frying pan" → "patelnia"
- Keep quantities, units and proper nouns unchanged
- Return only the translated strings, no explanations"""),
    ("human", "{input}"),
])

_translate_llm = ChatOllama(
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
    temperature=0.1,
    num_predict=4096,
    num_ctx=8192,
)


async def _translate_text(text: str) -> str:
    chain = _TRANSLATE_PROMPT | _translate_llm
    result = await chain.ainvoke({"input": text})
    return result.content.strip()


async def node_translate(state: AgentState) -> dict:
    log.info("node.translate.start")
    parsed = state["parsed"]

    if parsed is None:
        log.warning("node.translate.skip", reason="no parsed recipe")
        return {}

    import asyncio

    title_pl, description_pl = await asyncio.gather(
        _translate_text(parsed.title),
        _translate_text(parsed.description) if parsed.description else asyncio.sleep(0, result=None),
    )
    step_instructions_pl = await asyncio.gather(*[
        _translate_text(step.instruction)
        for step in parsed.steps
    ])
    unique_ingredients = list({ing.name for ing in parsed.ingredients})
    translated_ingredients = await asyncio.gather(*[
        _translate_text(name) for name in unique_ingredients
    ])
    ingredient_map = dict(zip(unique_ingredients, translated_ingredients))

    unique_items = list({
        item.name
        for step in parsed.steps
        for item in step.items
    })
    translated_items = await asyncio.gather(*[
        _translate_text(name) for name in unique_items
    ])
    item_map = dict(zip(unique_items, translated_items))

    translated_steps = []
    for step, instruction_pl in zip(parsed.steps, step_instructions_pl):
        translated_steps.append(
            step.model_copy(update={
                "instruction_i18n": {Language.PL: instruction_pl},
                "ingredients": [
                    ing.model_copy(update={
                        "name_i18n": {Language.PL: ingredient_map[ing.name]}
                    }) if hasattr(ing, "name_i18n") else ing
                    for ing in step.ingredients
                ],
                "items": [
                    item.model_copy(update={
                        "name_i18n": {Language.PL: item_map[item.name]}
                    }) if hasattr(item, "name_i18n") else item
                    for item in step.items
                ],
            })
        )

    result = ParsedRecipeWithTranslations(
        **parsed.model_dump(exclude={"steps"}),
        title_i18n={Language.PL: title_pl},
        description_i18n={Language.PL: description_pl} if description_pl else {},
        steps=translated_steps,
    )

    log.info("node.translate.ok", title_pl=title_pl)
    return {"parsed": result}



from langgraph.types import interrupt

def node_human_review(state: AgentState) -> dict:
    parsed = state["parsed"]

    summary = _build_summary(parsed)

    decision = interrupt({
        "summary": summary,
        "parsed": parsed.model_dump(),
        "message": "Sprawdź przepis i zdecyduj: save / skip / edit",
    })

    return {
        "human_decision": decision.get("action"),
        "human_note": decision.get("note"),
    }


def _build_summary(parsed: ParsedRecipeWithTranslations) -> str:
    lines = [
        f"TYTUŁ:    {parsed.title}",
        f"          {parsed.title_i18n.get('pl', '—')}",
        f"TRUDNOŚĆ: {parsed.difficulty_level or '—'}",
        f"CZAS:     {parsed.duration_minutes or '—'} min",
        f"KROKÓW:   {len(parsed.steps)}",
        f"SKŁADN.:  {len(parsed.ingredients)}",
        "",
        "KROKI:",
    ]
    for step in parsed.steps:
        pl = step.instruction_i18n.get("pl", "")
        lines.append(f"  {step.step_number}. {step.instruction}")
        if pl:
            lines.append(f"     ↳ {pl}")
        if step.ingredients:
            names = ", ".join(i.name for i in step.ingredients)
            lines.append(f"     + {names}")
    return "\n".join(lines)


from recipe_agent.db.supabase import get_client, save_full_recipe

async def node_save(state: AgentState) -> dict:
    log.info("node.save.start")
    sb = get_client()
    recipe_id = await save_full_recipe(sb, state["parsed"])
    log.info("node.save.ok", recipe_id=recipe_id)
    return {"saved_recipe_id": recipe_id}

from recipe_agent.db.supabase import update_recipe_i18n, save_steps

async def node_save_migration(state: AgentState) -> dict:
    log.info("node.save_migration.start", recipe_id=state.get("recipe_id"))
    sb = get_client()

    recipe_id = state["recipe_id"]
    parsed = state["parsed"]

    await update_recipe_i18n(sb, recipe_id, parsed)
    await save_steps(sb, recipe_id, parsed)

    log.info("node.save_migration.ok", recipe_id=recipe_id)
    return {"saved_recipe_id": recipe_id}