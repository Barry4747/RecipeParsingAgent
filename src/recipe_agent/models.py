from enum import StrEnum
from pydantic import BaseModel, Field


class Language(StrEnum):
    EN = "en"
    PL = "pl"


class ItemTag(StrEnum):
    BOWL = "bowl"
    POT = "pot"
    PAN = "pan"
    CUTLERY = "cutlery"
    MIXER = "mixer"
    BOARD = "board"
    KNIFE = "knife"
    OTHER = "other"


class DifficultyLevel(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class IngredientBase(BaseModel):
    name: str
    default_unit: str | None = None


class IngredientI18n(BaseModel):
    name_i18n: dict[Language, str] = Field(default_factory=dict)


class RecipeIngredient(BaseModel):
    name: str
    amount: float | None = None
    unit: str | None = None
    sort_order: int = 0


class ItemBase(BaseModel):
    name: str
    tag: ItemTag


class ItemI18n(BaseModel):
    name_i18n: dict[Language, str] = Field(default_factory=dict)


class StepIngredient(BaseModel):
    name: str
    amount: float | None = None
    unit: str | None = None
    name_i18n: dict[Language, str] = Field(default_factory=dict)


class StepItem(BaseModel):
    name: str
    tag: ItemTag
    name_i18n: dict[Language, str] = Field(default_factory=dict)


class RecipeStep(BaseModel):
    step_number: int
    instruction: str
    instruction_i18n: dict[Language, str] = Field(default_factory=dict)
    duration_seconds: int | None = None
    ingredients: list[StepIngredient] = Field(default_factory=list)
    items: list[StepItem] = Field(default_factory=list)


class ParsedRecipe(BaseModel):
    title: str
    description: str | None = None
    difficulty_level: DifficultyLevel | None = None
    duration_minutes: int | None = None
    category: str | None = None
    area: str | None = None
    tags: list[str] = Field(default_factory=list)
    source_url: str | None = None
    image_url: str | None = None
    youtube_url: str | None = None

    ingredients: list[RecipeIngredient] = Field(default_factory=list)
    steps: list[RecipeStep] = Field(default_factory=list)


class ParsedRecipeWithTranslations(ParsedRecipe):
    title_i18n: dict[Language, str] = Field(default_factory=dict)
    description_i18n: dict[Language, str] = Field(default_factory=dict)