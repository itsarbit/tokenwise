"""Pydantic data models for TokenWise."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RoutingStrategy(str, Enum):
    """Strategy for selecting a model."""

    CHEAPEST = "cheapest"
    BEST_QUALITY = "best_quality"
    BALANCED = "balanced"
    BUDGET_CONSTRAINED = "budget_constrained"


class ModelTier(str, Enum):
    """Model quality tier."""

    FLAGSHIP = "flagship"
    MID = "mid"
    BUDGET = "budget"


class ModelInfo(BaseModel):
    """Metadata and pricing for a single LLM."""

    id: str = Field(description="Model identifier, e.g. 'openai/gpt-4.1-mini'")
    name: str = Field(default="", description="Human-readable model name")
    provider: str = Field(default="", description="Provider name, e.g. 'OpenAI'")
    input_price: float = Field(default=0.0, description="Cost per million input tokens (USD)")
    output_price: float = Field(default=0.0, description="Cost per million output tokens (USD)")
    context_window: int = Field(default=4096, description="Maximum context length in tokens")
    capabilities: list[str] = Field(default_factory=list, description="e.g. ['code', 'reasoning']")
    tier: ModelTier = Field(default=ModelTier.BUDGET, description="Quality tier")

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for a given token count."""
        return (self.input_price * input_tokens + self.output_price * output_tokens) / 1_000_000


class Step(BaseModel):
    """A single step in a task plan."""

    id: int = Field(description="Step number (1-indexed)")
    description: str = Field(description="What this step does")
    model_id: str = Field(description="Assigned model ID")
    estimated_input_tokens: int = Field(default=500)
    estimated_output_tokens: int = Field(default=500)
    estimated_cost: float = Field(default=0.0, description="Estimated cost in USD")
    depends_on: list[int] = Field(default_factory=list, description="IDs of prerequisite steps")
    prompt_template: str = Field(default="", description="Prompt to send to the model")


class Plan(BaseModel):
    """A complete execution plan for a task."""

    task: str = Field(description="Original task description")
    steps: list[Step] = Field(default_factory=list)
    total_estimated_cost: float = Field(default=0.0)
    budget: float = Field(default=1.0, description="Budget cap in USD")

    def is_within_budget(self) -> bool:
        """Check if the plan's estimated cost fits the budget."""
        return self.total_estimated_cost <= self.budget


class StepResult(BaseModel):
    """Result of executing a single step."""

    step_id: int
    model_id: str
    output: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    actual_cost: float = 0.0
    success: bool = True
    error: str | None = None
    escalated: bool = Field(
        default=False, description="Whether this step was retried on a stronger model"
    )


class PlanResult(BaseModel):
    """Result of executing a full plan."""

    task: str
    step_results: list[StepResult] = Field(default_factory=list)
    total_cost: float = 0.0
    budget: float = 1.0
    success: bool = True
    final_output: str = ""

    @property
    def budget_remaining(self) -> float:
        return self.budget - self.total_cost


class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""

    role: str
    content: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = ConfigDict(extra="ignore")

    model: str = "auto"
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    tokenwise_opts: dict[str, Any] = Field(
        default_factory=dict,
        alias="tokenwise",
        description="TokenWise-specific options (strategy, budget)",
    )


class ChatCompletionChoice(BaseModel):
    """Single choice in a chat completion response."""

    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    """Token usage info."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class DeltaMessage(BaseModel):
    """Delta content in a streaming chunk."""

    role: str | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    """Single choice in a streaming chunk."""

    index: int = 0
    delta: DeltaMessage = Field(default_factory=DeltaMessage)
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: list[ChunkChoice] = Field(default_factory=list)
