import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from recipe_agent.graph.nodes import node_parse, node_translate, node_save
from recipe_agent.graph.state import AgentState


def base_state(**overrides) -> AgentState:
    return {
        "raw_text": "Pasta Carbonara recipe...",
        "source_url": None,
        "parsed": None,
        "parse_error": None,
        "human_decision": None,
        "human_note": None,
        "retry_count": 0,
        "messages": [],
        "saved_recipe_id": None,
        **overrides,
    }


@pytest.mark.asyncio
async def test_node_parse_success(sample_parsed_recipe):
    mock_result = {"parsed": sample_parsed_recipe, "parsing_error": None, "raw": ""}

    with patch("recipe_agent.graph.nodes._llm") as mock_llm:
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = mock_result
        mock_llm.__or__ = MagicMock(return_value=mock_chain)

        with patch("recipe_agent.graph.nodes._PARSE_PROMPT") as mock_prompt:
            mock_prompt.__or__ = MagicMock(return_value=mock_chain)

            result = await node_parse(base_state())

    assert result["parse_error"] is None
    assert result["parsed"].title == "Pasta Carbonara"
    assert result["retry_count"] == 1


@pytest.mark.asyncio
async def test_node_parse_appends_human_note():
    captured_input = {}

    async def fake_parse(text: str) -> None:
        captured_input["input"] = text
        raise ValueError("schema error")

    with patch("recipe_agent.graph.nodes._parse_with_retry", side_effect=fake_parse):
        await node_parse(base_state(human_note="Dodaj czas gotowania"))

    assert "Dodaj czas gotowania" in captured_input.get("input", "")


@pytest.mark.asyncio
async def test_node_parse_returns_error_after_retries():
    with patch("recipe_agent.graph.nodes._parse_with_retry",
               AsyncMock(side_effect=ValueError("bad json"))):
        result = await node_parse(base_state())

    assert result["parsed"] is None
    assert "bad json" in result["parse_error"]


@pytest.mark.asyncio
async def test_node_translate_fills_i18n(sample_parsed_recipe):
    translations = {
        "Pasta Carbonara": "Makaron Carbonara",
        "Classic Italian pasta dish": "Klasyczne włoskie danie",
        "Boil salted water and cook spaghetti until al dente": "Ugotuj makaron al dente",
        "Fry pancetta until crispy": "Podsmaż pancettę",
        "Mix eggs with parmesan in a bowl": "Wymieszaj jajka z parmezanem",
        "Combine pasta with pancetta and egg mixture off heat": "Połącz składniki",
        "spaghetti": "makaron spaghetti",
        "eggs": "jajka",
        "pancetta": "pancetta",
        "parmesan": "parmezan",
        "pot": "pot",
        "frying pan": "frying pan",
        "bowl": "bowl",
    }

    async def fake_translate(text: str) -> str:
        return translations.get(text, f"[{text}]")

    with patch("recipe_agent.graph.nodes._translate_text", side_effect=fake_translate):
        result = await node_translate(base_state(parsed=sample_parsed_recipe))

    translated = result["parsed"]
    assert translated.title_i18n["pl"] == "Makaron Carbonara"
    assert translated.steps[0].instruction_i18n["pl"] == "Ugotuj makaron al dente"


@pytest.mark.asyncio
async def test_node_translate_deduplicates_ingredients(sample_parsed_recipe):
    call_count = {}

    async def counting_translate(text: str) -> str:
        call_count[text] = call_count.get(text, 0) + 1
        return f"pl:{text}"

    with patch("recipe_agent.graph.nodes._translate_text", side_effect=counting_translate):
        await node_translate(base_state(parsed=sample_parsed_recipe))

    for name, count in call_count.items():
        assert count == 1, f"'{name}' przetłumaczono {count} razy"


@pytest.mark.asyncio
async def test_node_save_returns_recipe_id(sample_translated_recipe):
    with patch("recipe_agent.graph.nodes.get_client") as mock_get_client:
        with patch("recipe_agent.graph.nodes.save_full_recipe",
                   AsyncMock(return_value="uuid-1234")):
            result = await node_save(base_state(parsed=sample_translated_recipe))

    assert result["saved_recipe_id"] == "uuid-1234"