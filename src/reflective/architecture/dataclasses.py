from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Union

#################### Modules ###########################

@dataclass
class GuidelineRule:
    """
    A single expectation (norm/rule) a module evaluates against.

    Formerly ``GuidelineRule`` in the original script — fields are unchanged.
    """
    rule: str
    context: str = ""
    examples: List[str] = field(default_factory=list)
    fix_hint: str = ""    


@dataclass
class TurnContext:
    """
    Orchestrator-owned facts shared with every module for one turn.

    Contains only genuinely shared information (the user input, the
    conversation so far, and the scenario context). It is READ-ONLY during
    module execution: modules must not write to it, and modules must not
    communicate through it. Anything a module produces belongs in its
    ModuleVerdict (see ``artifact``).
    """
    user_input: str
    conversation_history: List[Dict]
    base_context: str


@dataclass
class ModuleVerdict:
    """
    The result of one module evaluating one candidate.

    score           - Normalized 0..1 score the framework's reconciler
                      compares against its threshold.
    passed          - Whether this module considers the candidate acceptable.
    classifications - Per-expectation labels in the MODULE'S OWN vocabulary
                      (e.g. SAFE / RISKY / VIOLATION for the EEC module).
                      The framework never interprets these; they are for
                      feedback and the trace.
    details         - Reasoning, raw evaluator output, and anything else the
                      module wants recorded in the reasoning trace.
    failures        - Structured failure records that feed the adapter
                      (prompt-revision) step on retry. Each entry should be a
                      dict the module's own ``feedback`` knows how to phrase.
    artifact        - The object this module's model component produced for
                      this candidate (e.g. the predicted conversation flow
                      for self-simulation). The orchestrator stores it under
                      ``candidate.annotations[module.name]``. None if the
                      module produces no artifact.
    """
    score: float
    passed: bool
    classifications: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)
    failures: List[Dict] = field(default_factory=list)
    artifact: Any = None


#################### Providers ###########################

@dataclass
class GenerationConfig:
    tokens: int = 512
    temperature: float = 0.9
    sampling: bool = True
    top_p: float = 0.9
    n_candidates: int = 2
    padding: Optional[int] = None


#################### Framework ###########################

@dataclass
class FrameworkConfig:
    max_attempts: int = 3
    min_score_threshold: float = 0.7
    debug_mode: bool = True


@dataclass
class ResponseCandidate:
    """One generated response. Verdicts and annotations are keyed by module name."""
    response: str
    annotations: Dict[str, Any] = field(default_factory=dict)
    verdicts: Dict[str, ModuleVerdict] = field(default_factory=dict)
    score: float = 0.0  # reconciled across modules
    generation_attempt: int = 1
    sequence_id: int = 1
    raw_llm_output: str = ""
    decision_reason: str = ""


@dataclass
class TurnResult:
    """The orchestrator's summary of one main_chat_handler run."""
    prompt: str
    final_response: str = ""
    attempts_used: int = 0
    success: bool = False
    fallback: bool = False
    fallback_reason: str = ""
    winning_score: Optional[float] = None
    winning_attempt: Optional[int] = None
    winning_sequence_id: Optional[int] = None
    all_candidates: List[ResponseCandidate] = field(default_factory=list)
    reasoning_trace: List[Dict] = field(default_factory=list)
    feedback_context: List[Dict] = field(default_factory=list)

    def winning_candidate(self) -> Optional[ResponseCandidate]:
        for c in self.all_candidates:
            if (c.generation_attempt == self.winning_attempt
                    and c.sequence_id == self.winning_sequence_id):
                return c
        return None

    def to_dict(self) -> Dict:
        data = asdict(self)
        # Derived, not stored: the winning candidate's per-module classifications.
        winning = self.winning_candidate()
        data["winning_classifications"] = (
            {name: v.classifications for name, v in winning.verdicts.items()}
            if winning else None
        )
        return data