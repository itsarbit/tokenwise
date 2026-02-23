"""Pydantic data models for TokenWise."""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class RoutingStrategy(str, Enum):
    """Strategy for selecting a model."""

    CHEAPEST = "cheapest"
    BEST_QUALITY = "best_quality"
    BALANCED = "balanced"


class ModelTier(str, Enum):
    """Model quality tier."""

    FLAGSHIP = "flagship"
    MID = "mid"
    BUDGET = "budget"


class EscalationPolicy(str, Enum):
    """How the executor handles fallback model selection."""

    FLEXIBLE = "flexible"
    MONOTONIC = "monotonic"


class TraceLevel(str, Enum):
    """Verbosity level for routing traces."""

    BASIC = "basic"
    VERBOSE = "verbose"


class TerminationState(str, Enum):
    """Final state of a routing/execution session."""

    COMPLETED = "completed"
    EXHAUSTED = "exhausted"
    ABORTED = "aborted"
    FAILED = "failed"
    NO_GO = "no_go"


class EscalationReasonCode(str, Enum):
    """Reason for escalating to a different model."""

    COMPLEXITY_THRESHOLD = "complexity_threshold"
    CAPABILITY_MISMATCH = "capability_mismatch"
    MODEL_ERROR = "model_error"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_GO = "no_go"
    MANUAL_OVERRIDE = "manual_override"


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
    required_capabilities: list[str] = Field(
        default_factory=list,
        description="Capabilities this step needs (e.g. ['code', 'reasoning'])",
    )


class Plan(BaseModel):
    """A complete execution plan for a task."""

    task: str = Field(description="Original task description")
    steps: list[Step] = Field(default_factory=list)
    total_estimated_cost: float = Field(default=0.0)
    budget: float = Field(default=1.0, description="Budget cap in USD")
    decomposition_source: str = Field(
        default="llm", description="How steps were produced: 'llm' or 'fallback'"
    )
    decomposition_error: str | None = Field(
        default=None, description="Error message if decomposition fell back"
    )
    planner_cost: float = Field(default=0.0, description="Cost of the LLM decomposition call")

    def is_within_budget(self) -> bool:
        """Check if the plan's estimated cost fits the budget."""
        return self.total_estimated_cost <= self.budget


class LedgerEntry(BaseModel):
    """A single cost record in the execution ledger."""

    reason: str = Field(description="e.g. 'step 1 attempt 1' or 'step 1 escalation attempt 2'")
    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    success: bool = True


class CostLedger(BaseModel):
    """Tracks all spend across attempts and escalations."""

    entries: list[LedgerEntry] = Field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(e.cost for e in self.entries)

    @property
    def wasted_cost(self) -> float:
        return sum(e.cost for e in self.entries if not e.success)

    def record(
        self,
        reason: str,
        model_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        success: bool = True,
    ) -> None:
        self.entries.append(
            LedgerEntry(
                reason=reason,
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                success=success,
            )
        )


class EscalationRecord(BaseModel):
    """A single escalation event during execution."""

    from_model: str
    from_tier: ModelTier
    to_model: str
    to_tier: ModelTier
    reason_code: EscalationReasonCode
    step_id: int | None = None
    trigger_score: float | None = None
    threshold_value: float | None = None


class RoutingTrace(BaseModel):
    """Structured trace of routing decisions."""

    request_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    initial_model: str = ""
    initial_tier: ModelTier = ModelTier.BUDGET
    escalation_policy: EscalationPolicy = EscalationPolicy.FLEXIBLE
    escalations: list[EscalationRecord] = Field(default_factory=list)
    final_model: str = ""
    final_tier: ModelTier = ModelTier.BUDGET
    termination_state: TerminationState = TerminationState.COMPLETED
    budget_used: float = 0.0
    budget_remaining: float = 0.0
    trace_level: TraceLevel = TraceLevel.BASIC


class RiskGateBlockedError(Exception):
    """Raised when the risk gate blocks a query."""

    def __init__(self, reason: str, trace: RoutingTrace | None = None) -> None:
        self.reason = reason
        self.trace = trace
        super().__init__(reason)


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
    http_status_code: int | None = Field(
        default=None, description="HTTP status code from the provider, if the error was HTTP-based"
    )
    escalated: bool = Field(
        default=False, description="Whether this step was retried on a stronger model"
    )
    routing_trace: RoutingTrace | None = None


class PlanResult(BaseModel):
    """Result of executing a full plan."""

    task: str
    step_results: list[StepResult] = Field(default_factory=list)
    skipped_steps: list[Step] = Field(
        default_factory=list,
        description="Steps that were not executed (e.g. budget exhausted)",
    )
    total_cost: float = 0.0
    budget: float = 1.0
    success: bool = True
    final_output: str = ""
    ledger: CostLedger = Field(default_factory=CostLedger)
    routing_trace: RoutingTrace | None = None

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
    tokenwise_trace: dict[str, Any] | None = None


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
