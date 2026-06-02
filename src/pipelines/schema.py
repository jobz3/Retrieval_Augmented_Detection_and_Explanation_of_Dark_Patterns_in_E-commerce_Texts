"""
Structured output schema for the dark pattern explanation pipeline.

Every LLM call must return a dict matching PredictionResult.
The JSON schema is also used in prompts so the model knows the required format.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PatternType(str, Enum):
    SCARCITY = "Scarcity"
    URGENCY = "Urgency"
    SOCIAL_PROOF = "Social Proof"
    MISDIRECTION = "Misdirection"
    OBSTRUCTION = "Obstruction"
    FORCED_ACTION = "Forced Action"
    SNEAKING = "Sneaking"
    NONE = "Not Dark Pattern"
    # Internal sentinel for parse / LLM-call failures only. It is NOT offered to
    # the model as a label choice (the prompt and OUTPUT_SCHEMA_STR list only the
    # eight categories above); the model signals doubt via a low confidence value.
    UNCERTAIN = "Uncertain"


class PredictionResult(BaseModel):
    """
    Structured output produced by the LLM for a single product text.

    Fields:
        reasoning_steps:        Ordered list of reasoning steps before the final label.
        label:                  One of the eight categories (a dark-pattern type or
                                "Not Dark Pattern"). "Uncertain" is an internal
                                failure sentinel only, never a model-chosen label.
        confidence:             Self-assessed probability [0, 1] for the label.
        psychological_mechanism: Named cognitive bias or persuasion principle.
        evidence_span:          Verbatim substring from the input text.
        harm_dimension:         Short phrase describing consumer harm.
        rationale:              1–2 sentence natural language explanation.
        rewrite:                Detoxified version of the full product text.
    """

    reasoning_steps: list[str] = Field(
        default_factory=list,
        description="Ordered chain-of-thought steps leading to the label (populate before label)"
    )
    label: PatternType = Field(description="Dark pattern category or 'Not Dark Pattern'")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in the label [0,1]")
    psychological_mechanism: str = Field(
        description="Named cognitive bias or persuasion principle, e.g. 'loss aversion'"
    )
    evidence_span: str = Field(
        description="Exact verbatim substring from the input text that triggers the pattern"
    )
    harm_dimension: str = Field(
        description="Short phrase describing the potential consumer harm, e.g. 'false urgency'"
    )
    rationale: str = Field(
        description="1-2 sentence explanation of why this text constitutes the identified pattern"
    )
    rewrite: str = Field(
        description="Full product text rewritten to remove the dark pattern while preserving the core offer"
    )

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)

    def is_dark_pattern(self) -> bool:
        return self.label not in (PatternType.NONE, PatternType.UNCERTAIN)

    def span_is_grounded(self, input_text: str) -> bool:
        """Check that evidence_span appears verbatim in the input text."""
        return self.evidence_span.strip() in input_text

    def to_dict(self) -> dict:
        d = self.model_dump()
        d["label"] = self.label.value
        return d


# JSON schema string to embed in prompts
OUTPUT_SCHEMA_STR = """{
  "reasoning_steps": [
    "<step 1: identify any manipulation tactics present in the text>",
    "<step 2: match tactics to the most fitting dark pattern category>",
    "<step 3: assess confidence and note any ambiguities>"
  ],
  "label": "<one of: Scarcity | Urgency | Social Proof | Misdirection | Obstruction | Forced Action | Sneaking | Not Dark Pattern>",
  "confidence": <float between 0.0 and 1.0>,
  "psychological_mechanism": "<name of cognitive bias or persuasion principle>",
  "evidence_span": "<verbatim substring from the input text>",
  "harm_dimension": "<short phrase describing consumer harm>",
  "rationale": "<1-2 sentence explanation>",
  "rewrite": "<full product text with dark pattern removed, core offer preserved>"
}"""
