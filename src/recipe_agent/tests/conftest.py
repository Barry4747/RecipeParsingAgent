import pytest
from unittest.mock import AsyncMock, patch
from recipe_agent.models import (
    ParsedRecipe, ParsedRecipeWithTranslations,
    RecipeIngredient, RecipeStep, StepIngredient, StepItem,
    ItemTag, Language, IngredientAction,
)


@pytest.fixture
def sample_parsed_recipe() -> ParsedRecipe:
    return ParsedRecipe(
        title="Pasta Carbonara",
        description="Classic Italian pasta dish",
        difficulty_level="medium",
        duration_minutes=30,
        category="Pasta",
        area="Italian",
        tags=["pasta", "italian", "quick"],
        ingredients=[
            RecipeIngredient(name="spaghetti", amount=400, unit="g", sort_order=0),
            RecipeIngredient(name="eggs", amount=4, unit=None, sort_order=1),
            RecipeIngredient(name="pancetta", amount=150, unit="g", sort_order=2),
            RecipeIngredient(name="parmesan", amount=100, unit="g", sort_order=3),
        ],
        steps=[
            RecipeStep(
                step_number=1,
                instruction="Boil salted water and cook spaghetti until al dente",
                duration_seconds=600,
                ingredients=[StepIngredient(name="spaghetti", amount=400, unit="g", actions=[IngredientAction.ADD])],
                items=[StepItem(name="pot", tag=ItemTag.POT)],
            ),
            RecipeStep(
                step_number=2,
                instruction="Fry pancetta until crispy",
                duration_seconds=300,
                ingredients=[StepIngredient(name="pancetta", amount=150, unit="g", actions=[IngredientAction.ADD])],
                items=[StepItem(name="frying pan", tag=ItemTag.PAN)],
            ),
            RecipeStep(
                step_number=3,
                instruction="Mix eggs with parmesan in a bowl",
                ingredients=[
                    StepIngredient(name="eggs", amount=4, actions=[IngredientAction.ADD]),
                    StepIngredient(name="parmesan", amount=100, unit="g", actions=[IngredientAction.GRATE, IngredientAction.ADD]),
                ],
                items=[StepItem(name="bowl", tag=ItemTag.BOWL)],
            ),
            RecipeStep(
                step_number=4,
                instruction="Combine pasta with pancetta and egg mixture off heat",
                ingredients=[],
                items=[],
            ),
        ],
    )


@pytest.fixture
def sample_translated_recipe(sample_parsed_recipe) -> ParsedRecipeWithTranslations:
    translated_steps = []
    for step in sample_parsed_recipe.steps:
        translated_steps.append(
            step.model_copy(update={
                "instruction_i18n": {Language.PL: f"Przetłumaczona instrukcja {step.step_number}"},
            })
        )
    return ParsedRecipeWithTranslations(
        **sample_parsed_recipe.model_dump(exclude={"steps"}),
        title_i18n={Language.PL: "Makaron Carbonara"},
        description_i18n={Language.PL: "Klasyczne włoskie danie z makaronem"},
        steps=translated_steps,
    )