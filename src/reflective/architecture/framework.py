"""
Reflective framework orchestrator.

The generic reflective loop from the original script, with:
  - Provider injected (no globals).
  - Modules injected and fanned out per candidate (each module is independent
    and returns a ModuleVerdict; the framework never interprets module
    vocabulary).
  - reconcile(): the paper's Reconciler. Current implementation is
    pessimistic — any module objecting fails the candidate.
  - Three result tiers: ModuleVerdict (module x candidate) inside
    ResponseCandidate (one generated response) inside TurnResult (one turn).

Generation is module-agnostic: modules may contribute prompt fragments via
generate() and may parse the generator's raw output into a
(response, artifact) pair via parse_generation(). The orchestrator contains
no module vocabulary.
"""

###################### Imports ######################
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Union

# Package logger. NullHandler = quiet by default when embedded in a host
# application; the experiment runner (or the host) attaches real handlers.
logging.getLogger("reflective").addHandler(logging.NullHandler())
logger = logging.getLogger("reflective.framework")

from ..providers.provider_base import Provider
from ..modules.module_base import ReflectiveModule 
from .dataclasses import ModuleVerdict, TurnContext, FrameworkConfig, TurnResult, ResponseCandidate


###################### Default Slots ######################
def default_feedback_slot(feedback_context: List[Dict]) -> str:
    if not feedback_context:
        return ""
    section = "\n\nIMPORTANT - DON'T VIOLATE THESE GUIDELINES:\n"
    for i, f in enumerate(feedback_context):
        # "Flagged by" identifies which module flagged the failure.
        section += (
            f"\nNumber {i+1}:\n"
            f"- Guideline: {f['rule']}\n"
            f"- Flagged by: {f.get('module', 'unknown')}\n"
            f"- How to Avoid: {f.get('fix_hint') or f['context']}\n"
            f"- Previous Sequence ID: {f.get('sequence_id', 'N/A')}\n"
        )
    section += "\nMake sure ALL your response variations avoid these issues!\n"
    return section  


def default_attempt_guidance_slot(attempt: int) -> str:
    if attempt == 2:
        return (
            "\n\nATTEMPT 2 GUIDANCE:\n"
            "- Try different approaches that follow the guidelines\n"
            "- Consider how conversations might evolve differently\n"
            "- Generate diverse response styles while respecting guidelines\n"
        )
    if attempt >= 3:
        return (
            f"\n\nATTEMPT {attempt} GUIDANCE:\n"
            "- This is your final attempt, be very careful about guidelines\n"
            "- Consider conservative strategies\n"
            "- Prioritize guideline compliance over creativity\n"
        )
    return ""


DEFAULT_INSTRUCTIONS = (
    "Generate a response to the user's question.\n\n"
    "Your response should be natural and helpful while considering "
    "the context and any guidance provided above."
)

DEFAULT_FOOTER = "IMPORTANT: Generate diverse, varied responses while following all guidelines."

DEFAULT_SLOTS: Dict[str, Union[None, str, Callable]] = {
    "feedback": default_feedback_slot,              # callable(feedback_context) -> str
    "attempt_guidance": default_attempt_guidance_slot,  # callable(attempt) -> str
    "instructions": None,   # None -> module build_generation_prompt()s, else DEFAULT_INSTRUCTIONS
    "footer": DEFAULT_FOOTER,
}


###################### Orchestrator ######################
class AgenticFramework:
    def __init__(
        self,
        provider: Provider,
        modules: List[ReflectiveModule],
        base_context: str,
        framework_config: Optional[FrameworkConfig] = None,
        slots: Optional[Dict[str, Union[None, str, Callable]]] = None,
    ):
        if not modules:
            logger.warning("Warning: no modules provided.")
        if not base_context:
            logger.warning("Warning: empty base_context.")

        unknown = set(slots or {}) - set(DEFAULT_SLOTS)
        if unknown:
            raise ValueError(f"Unknown prompt slots: {sorted(unknown)}. "
                             f"Available: {sorted(DEFAULT_SLOTS)}")

        self.provider = provider
        self.modules = modules
        self.base_context = base_context
        self.framework_config = framework_config or FrameworkConfig()
        self.slots = {**DEFAULT_SLOTS, **(slots or {})}
        # Read once; every later use goes through these attributes.
        self.max_attempts = self.framework_config.max_attempts
        self.min_score_threshold = self.framework_config.min_score_threshold
        self.debug_mode = self.framework_config.debug_mode

    def main_chat_handler(self, user_input: str, conversation_history: List[Dict]) -> TurnResult:
        logger.info(f"\n{'='*50}")
        logger.info(f"USER INPUT: {user_input}")
        logger.info(f"{'='*50}")

        # The per-turn projection of the paper's Updated State of the World.
        # Read-only during module execution; also the single source of truth
        # for the generation prompt.
        ctx = TurnContext(
            user_input=user_input,
            conversation_history=conversation_history,
            base_context=self.base_context,
        )

        feedback_context: List[Dict] = []
        all_candidates: List[ResponseCandidate] = []
        result = TurnResult(prompt=user_input)

        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"\nATTEMPT {attempt}/{self.max_attempts}")
            result.attempts_used = attempt

            candidates = self.generate_candidates(ctx, feedback_context, attempt)
            if not candidates:
                logger.info("Failed to generate candidates")
                continue

            attempt_failures: List[Dict] = []

            for candidate in candidates:
                candidate.generation_attempt = attempt
                all_candidates.append(candidate)

                # Independent module fan-out (parallel-ready; sequential for now).
                verdicts = {m.name: m.run(ctx, candidate, self.provider) for m in self.modules}
                candidate.verdicts = verdicts
                for name, v in verdicts.items():
                    if v.artifact is not None:
                        candidate.annotations[name] = v.artifact

                candidate.score = self.reconcile(verdicts)


                if candidate.score >= self.min_score_threshold:
                    candidate.decision_reason = (
                        f"Sequence {candidate.sequence_id} with score "
                        f"{candidate.score} meets threshold"
                    )
                    result.reasoning_trace.append(self._trace_entry(candidate))
                    self._print_debug(candidate)
                    logger.info(
                        f"\n🎉 SUCCESS: Accepted sequence {candidate.sequence_id} "
                        f"on attempt {attempt}"
                    )
                    result.final_response = candidate.response
                    result.success = True
                    result.winning_score = candidate.score
                    result.winning_attempt = attempt
                    result.winning_sequence_id = candidate.sequence_id
                    break
                else:
                    candidate.decision_reason = (
                        f"Sequence {candidate.sequence_id} failed threshold "
                        f"({self.min_score_threshold})"
                    )
                    for m in self.modules:
                        attempt_failures.extend(m.feedback(verdicts[m.name]))

                    result.reasoning_trace.append(self._trace_entry(candidate))
                    self._print_debug(candidate)

            if result.success:
                break

            if attempt_failures:
                self._update_feedback_context(feedback_context, attempt_failures)
                result.reasoning_trace.append({
                    "event": "context_update",
                    "attempt": attempt,
                    "added_failures": attempt_failures,
                })
                logger.info(
                    f"\nUPDATING CONTEXT: Added {len(attempt_failures)} "
                    f"failed expectations"
                )

        result.all_candidates = all_candidates

        # Integrator fallback: pick least-bad if no candidate cleared threshold.
        if not result.success:
            logger.info(f"\nMAX ATTEMPTS REACHED. Selecting least-bad from {len(all_candidates)} candidates.")
            if all_candidates:
                best = self._select_least_bad(all_candidates)
                result.final_response = best.response
                result.winning_score = best.score
                result.winning_attempt = best.generation_attempt
                result.winning_sequence_id = best.sequence_id
                result.fallback = True
                result.fallback_reason = (
                    f"No candidate reached threshold {self.min_score_threshold}; "
                    f"selected sequence {best.sequence_id} from attempt "
                    f"{best.generation_attempt} with score {best.score}"
                )
            else:
                result.final_response = "I'm having trouble processing your request right now."

        result.feedback_context = feedback_context
        return result

    def reconcile(self, verdicts: Dict[str, ModuleVerdict]) -> float:
        """

        The candidate's score is the worst module score, so any module objecting
        fails the candidate. Also the future home of discrepancy flagging
        and targeted re-evaluation.
        """
        if not verdicts:
            return 1.0
        return min(v.score for v in verdicts.values())

    def generate_candidates(
        self,
        ctx: TurnContext,
        feedback_context: Optional[List[Dict]] = None,
        attempt: int = 1,
    ) -> List[ResponseCandidate]:
        logger.info(f"\nGENERATING CANDIDATES (Attempt {attempt})")
        feedback_context = feedback_context or []

        history_text = ""
        for msg in ctx.conversation_history[-5:]:
            history_text += f"{msg['role'].upper()}: {msg['content']}\n"

        failed_section = self.build_slot("feedback", feedback_context)
        attempt_guidance = self.build_slot("attempt_guidance", attempt)

        # An explicit "instructions" slot wins; otherwise each module may
        # contribute directives via build_generation_prompt() (e.g.
        # self-simulation's flow-prediction format); with neither, the
        # generator is simply asked for a response and the raw output is used.
        instruction_block = self.build_slot("instructions")
        if instruction_block is None:
            directives = [d for m in self.modules if (d := m.build_generation_prompt(ctx))]
            instruction_block = "\n\n".join(directives) if directives else DEFAULT_INSTRUCTIONS

        prompt = f"""CONTEXT: {ctx.base_context}

CONVERSATION HISTORY:
{history_text}

USER JUST ASKED: {ctx.user_input}

{failed_section}{attempt_guidance}

{instruction_block}

{self.build_slot("footer")}"""

        llm_responses = self.provider.generate([{"role": "user", "content": prompt}])
        candidates: List[ResponseCandidate] = []

        for i, raw in enumerate(llm_responses):
            logger.info(f"\nPARSING SEQUENCE {i+1}/{len(llm_responses)}")

            # Modules that emitted generation directives know the output
            # format; give each a chance to parse its artifact. The first
            # module to supply a response text wins; otherwise the raw
            # output is the response.
            response_text: Optional[str] = None
            artifacts: Dict[str, Any] = {}
            for m in self.modules:
                parsed = m.parse_generation(raw)
                if parsed:
                    override, artifact = parsed
                    if artifact is not None:
                        artifacts[m.name] = artifact
                    if override and response_text is None:
                        response_text = override

            if response_text is None:
                response_text = raw.strip()

            if response_text:
                candidate = ResponseCandidate(response_text)
                candidate.annotations.update(artifacts)
                candidate.sequence_id = i + 1
                candidate.raw_llm_output = raw
                candidates.append(candidate)
                logger.info(f"Sequence {i+1}: {candidate.response[:50]}...")
            else:
                logger.info(f"Failed to parse sequence {i+1}")

        return candidates

    def build_slot(self, slot: str, *args):
        """A slot's content for this call: strings pass through, callables
        are invoked with the runtime state that slot depends on."""
        value = self.slots[slot]
        return value(*args) if callable(value) else value


    def _select_least_bad(self, candidates: List[ResponseCandidate]) -> ResponseCandidate:
        ranked = sorted(
            candidates,
            key=lambda x: (-x.score, x.generation_attempt, x.sequence_id),
        )
        best = ranked[0]
        logger.info(
            f"\nFALLBACK: Sequence {best.sequence_id}, score {best.score}, "
            f"attempt {best.generation_attempt}"
        )
        return best

    # Added a feedback key to idenify which module is providing which feedback string.
    @staticmethod
    def _feedback_key(record: Dict) -> tuple:
        """Identity of a feedback record: same expectation, same observing module."""
        return (record.get("module", ""), record.get("rule", ""))

    def _update_feedback_context(self, existing: List[Dict], new: List[Dict]) -> None:
        for nf in new:
            for i, ef in enumerate(existing):
                if self._feedback_key(ef) == self._feedback_key(nf):
                    existing[i] = nf          # replace, don't merge
                    break
            else:
                existing.append(nf)

    @staticmethod
    def _trace_entry(candidate: ResponseCandidate) -> Dict:
        """Structured snapshot of the framework's reasoning for one candidate."""
        return {
            "event": "candidate_evaluation",
            "attempt": candidate.generation_attempt,
            "sequence_id": candidate.sequence_id,
            "response": candidate.response,
            "raw_llm_output": candidate.raw_llm_output,
            "score": candidate.score,
            "verdicts": {name: asdict(v) for name, v in candidate.verdicts.items()},
            "decision_reason": candidate.decision_reason,
        }

    def _print_debug(self, candidate: ResponseCandidate) -> None:
        if not self.debug_mode:
            return
        logger.debug(f"\nDEBUG - Attempt {candidate.generation_attempt}, Sequence {candidate.sequence_id}")
        logger.debug(f"   Response: {candidate.response[:120]}...")
        logger.debug(f"   Reconciled score: {candidate.score}")
        for name, v in candidate.verdicts.items():
            mark = "OK" if v.passed else "X"
            logger.debug(f"     {mark} [{name}] {v.classifications} (score {v.score})")
            for f in v.failures:
                logger.debug(f"         - {f.get('rule', '')[:40]}: {f.get('classification', '')} "
                      f"- {f.get('reasoning', '')[:80]}")
        logger.debug(f"   Decision: {candidate.decision_reason}")