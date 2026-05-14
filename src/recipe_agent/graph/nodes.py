import json
import structlog
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field
from recipe_agent.config import settings
from recipe_agent.models import ParsedRecipe, ParsedRecipeWithTranslations, Language, RecipeStep
from recipe_agent.graph.state import AgentState

log = structlog.get_logger()

_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0,
    api_key=settings.google_api_key,
).with_structured_output(ParsedRecipe, include_raw=True)

_PARSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a culinary recipe parser. Extract structured data from recipe text.

CRITICAL INSTRUCTION - STEP SPLITTING:
You MUST break down the original recipe text into VERY SHORT micro-steps. 
Do NOT just copy original paragraphs. 1 Step = 1 Single Action!
BAD example (do not do this): "Heat oven to 180C. Put oats and flour in a bowl. Melt butter."
GOOD example:
Step 1: "Heat oven to 180C."
Step 2: "Put oats and flour in a bowl."
Step 3: "Melt butter."

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

Tool tags — assign the CLOSEST match, default to "other" when unsure:
- "bowl" -> bowl, mixing bowl, salad bowl
- "pot" -> pot, saucepan, stockpot, Dutch oven
- "pan" -> frying pan, skillet, wok, griddle
- "cutlery" -> spoon, fork, spatula, whisk, ladle, tongs
- "mixer" -> electric mixer, stand mixer, blender, food processor
- "board" -> cutting board, chopping board
- "knife" -> knife, chef's knife, paring knife
- "other" -> oven, baking sheet, rolling pin, wire rack, grater, peeler, timer, pan (everything else)

IMPORTANT: "baking sheet", "wire rack", "oven", "rolling pin" -> always "other"
IMPORTANT: "bowl" -> always "bowl", never "mixer"

Rules for ingredients list (recipe-level):
- Only include ingredients that need to be PURCHASED
- Do NOT include intermediate products created during cooking

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

    if result.get("parsing_error"):
        raise ValueError(f"Schema mismatch: {result['parsing_error']}")

    parsed = result.get("parsed")
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


class TranslationResult(BaseModel):
    title_pl: str
    description_pl: str | None
    steps_pl: list[str] = Field(description="Przetłumaczone kroki w tej samej kolejności co w oryginale")
    ingredients_map: dict[str, str] = Field(description="Słownik: nazwa angielska -> nazwa polska (mianownik)")
    items_map: dict[str, str] = Field(description="Słownik: nazwa angielska -> nazwa polska (mianownik)")

_TRANSLATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a culinary translator specializing in Polish.
Translate the provided JSON data from English to Polish.
Rules:
- Keep culinary terms accurate (e.g. "fold" -> "delikatnie wmieszaj")
- Ingredient names MUST be in nominative case (mianownik): "onion" -> "cebula", "eggs" -> "jajka"
- Tool names MUST be in nominative case: "frying pan" -> "patelnia"
- Keep quantities and units unchanged
- Return strictly the requested JSON structure."""),
    ("human", "Translate this recipe data:\n{input_data}"),
])

_translate_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0.1,
    api_key=settings.google_api_key,
).with_structured_output(TranslationResult)

async def node_translate(state: AgentState) -> dict:
    log.info("node.translate.start")
    parsed = state["parsed"]

    if parsed is None:
        return {}

    # Zbieramy wszystkie unikalne składniki i narzędzia do przetłumaczenia
    unique_ingredients = list({ing.name for ing in parsed.ingredients} | {ing.name for step in parsed.steps for ing in step.ingredients})
    unique_items = list({item.name for step in parsed.steps for item in step.items})
    steps_en = [step.instruction for step in parsed.steps]

    # Budujemy paczkę (JSON) dla modelu
    payload = {
        "title": parsed.title,
        "description": parsed.description or "",
        "steps": steps_en,
        "ingredients_to_translate": unique_ingredients,
        "items_to_translate": unique_items
    }

    # Wywołujemy model RAZ
    chain = _TRANSLATE_PROMPT | _translate_llm
    translations: TranslationResult = await chain.ainvoke({"input_data": json.dumps(payload)})

    # Składamy wszystko z powrotem do ParsedRecipeWithTranslations
    translated_steps = []
    for step, instruction_pl in zip(parsed.steps, translations.steps_pl):
        translated_steps.append(
            step.model_copy(update={
                "instruction_i18n": {Language.PL: instruction_pl},
                "ingredients": [
                    ing.model_copy(update={
                        "name_i18n": {Language.PL: translations.ingredients_map.get(ing.name, ing.name)}
                    }) if hasattr(ing, "name_i18n") else ing
                    for ing in step.ingredients
                ],
                "items": [
                    item.model_copy(update={
                        "name_i18n": {Language.PL: translations.items_map.get(item.name, item.name)}
                    }) if hasattr(item, "name_i18n") else item
                    for item in step.items
                ],
            })
        )

    result = ParsedRecipeWithTranslations(
        **parsed.model_dump(exclude={"steps"}),
        title_i18n={Language.PL: translations.title_pl},
        description_i18n={Language.PL: translations.description_pl} if translations.description_pl else {},
        steps=translated_steps,
    )

    log.info("node.translate.ok", title_pl=translations.title_pl)
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
        f"TRUDNOŚĆ: {parsed.difficulty_level or '—'}",
        f"CZAS:     {parsed.duration_minutes or '—'} min",
        f"KROKÓW:   {len(parsed.steps)}",
        f"SKŁADN.:  {len(parsed.ingredients)}",
        "",
        "KROKI:",
    ]
    for step in parsed.steps:
        lines.append(f"  {step.step_number}. {step.instruction}")
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