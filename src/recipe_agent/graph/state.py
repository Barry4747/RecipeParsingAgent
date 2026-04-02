from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from recipe_agent.models import ParsedRecipe, ParsedRecipeWithTranslations


class AgentState(TypedDict, total=False):
    raw_text: str
    source_url: str | None
    recipe_id: str | None

    parsed: ParsedRecipe | ParsedRecipeWithTranslations | None
    parse_error: str | None

    human_decision: str | None
    human_note: str | None

    retry_count: int
    messages: Annotated[list, add_messages]

    saved_recipe_id: str | None