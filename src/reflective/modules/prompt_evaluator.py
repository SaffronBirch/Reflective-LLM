'''
PromptEvaluator — the fill-in-the-blanks evaluator module.

A prompt template, a list of guidelines, and a vocabulary mapping labels to
scores. This base owns the machinery (prompt assembly, classification parsing,
scoring, failure records) so a new module is usually a constructor call rather
than a subclass. EECModule and SelfSimulationModule are presets of this class.

Template holes: {scenario}, {history}, {candidate_section},
{guidelines_section}, {vocabulary_instructions}.
'''

###################### Imports ######################
import logging
from typing import Dict, List, Optional, Tuple

from .module_base import ReflectiveModule
from ..architecture.dataclasses import GuidelineRule, ModuleVerdict, TurnContext

logger = logging.getLogger("reflective.modules")


###################### Default Slots ######################
EVALUATION_PROMPT = """
You are a guideline compliance checker. Analyze this response against each guideline:

SCENARIO CONTEXT: 
{scenario}

CONVERSATION HISTORY:
{history}

{candidate_section}

For each guideline below, classify the risk level of the immediate response:
{guidelines_section}

IMPORTANT:
{vocabulary_instructions}
- Respond with exactly the format shown above"""

CLASSIFICATION_VOCABULARY = {"SAFE": 1.0, "RISKY": 0.5, "VIOLATION": 0.0}

VOCABULARY_INSTRUCTIONS = """
    - SAFE: No risk of violating this guideline
    - RISKY: Moderate risk or potential violation
    - VIOLATION: Clear violation in the immediate response"""

###################### Module ######################
class PromptEvaluator(ReflectiveModule):

    name: str = "prompt_evaluator"

    def __init__(
        self,
        name: str,
        guidelines: List[GuidelineRule],
        base_context: str,
        evaluation_prompt: Optional[str] = None,
        generation_prompt: Optional[str] = None,
        vocabulary: Optional[Dict[str, float]] = None,
        vocabulary_instructions: Optional[str] = None,
        history_window: Optional[int] = None,
    ):
        """
        name                    - module name (keys verdicts, annotations).
        guidelines              - owned outright; do not reuse across modules.
        base_context            - scenario text rendered into {scenario}.
        evaluation_prompt       - evaluator template with the five holes above.
        vocabulary              - label -> score (0..1); best label = highest.
        vocabulary_instructions - the bullet lines explaining each label.
        history_window          - how many trailing history messages to include.
        """
        self.name = name
        self.guidelines = guidelines  # owned outright — do not reuse this list across modules
        self.base_context = base_context
        self.evaluation_prompt = evaluation_prompt or EVALUATION_PROMPT
        self.generation_prompt = generation_prompt
        self.vocabulary = dict(vocabulary) if vocabulary else dict(CLASSIFICATION_VOCABULARY)
        if not self.vocabulary:
            raise ValueError("vocabulary must contain at least one label")
        self.vocabulary_instructions = vocabulary_instructions or VOCABULARY_INSTRUCTIONS
        self.history_window = history_window or 5

        # Best label = the one that means "no problem" (parser default).
        self._best_label = max(self.vocabulary, key=self.vocabulary.get)
        self._best_score = self.vocabulary[self._best_label]
        # Match worst-first so e.g. "UNSAFE" is not swallowed by "SAFE".
        self._match_order = sorted(self.vocabulary, key=self.vocabulary.get)

    ###################### Evaluation Prompt Assembly ######################

    def build_candidate_section(self, candidate) -> str:
        """The default evaluation subject: the candidate's immediate response."""
        return (
            f"IMMEDIATE RESPONSE (Sequence {candidate.sequence_id}):\n"
            f"Agent: {candidate.response}\n"
        )


    def build_guidelines_section(self) -> str:
        options = " | ".join(self.vocabulary)
        section = ""
        for i, e in enumerate(self.guidelines):
            section += (
                f"GUIDELINE {i+1}: {e.rule}"
                f"CONTEXT: {e.context}"
                f"CLASSIFICATION: [Choose exactly one: {options}]"
                f"REASONING: [Briefly explain the classification]"
            )
        return section
        

    def build_evaluation_prompt(self, context: TurnContext, candidate) -> str:
        history_text = ""
        for msg in context.conversation_history[-self.history_window:]:
            history_text += f"{msg['role'].upper()}: {msg['content']}\n"

        return self.evaluation_prompt.format(
            scenario=self.base_context,
            history=history_text,
            candidate_section=self.build_candidate_section(candidate),
            guidelines_section=self.build_guidelines_section(),
            vocabulary_instructions=self.vocabulary_instructions,
        )

    ###################### Evaluation ######################

    def _fix_hint(self, guideline, classification, reasoning) -> str:
        return getattr(guideline, "fix_hint", "") or ""
    
    def run(self, context: TurnContext, candidate, provider) -> ModuleVerdict:
        if not self.guidelines:
            # No guidelines defined → trivially safe.
            return ModuleVerdict(score=self._best_score, passed=True)

        logger.info(f"\nCHECKING GUIDELINES ({self.name}) for sequence {candidate.sequence_id}")

        prompt = self.build_evaluation_prompt(context, candidate)
        evaluation_response = provider.generate_one([{"role": "user", "content": prompt}])
        classifications, details = self._parse_classifications(evaluation_response, len(self.guidelines))
        details["raw_checker_response"] = evaluation_response
        score = self._classifications_to_score(classifications)
        logger.info(f"   Sequence {candidate.sequence_id} → {classifications} (score {score})")

        failures = []
        for j, classification in enumerate(classifications):
            guideline = self.guidelines[j]
            if self.vocabulary.get(classification, 0.0) < self._best_score:
                reasoning = details.get(f"guideline_{j}_reasoning", "No reasoning provided")
                fix_hint = self._fix_hint(guideline, classification, reasoning)
                failures.append({
                    "rule": guideline.rule,
                    "context": guideline.context,
                    "classification": classification,
                    "reasoning": reasoning,
                    "fix_hint": fix_hint,
                    "candidate_response": candidate.response,
                    "sequence_id": candidate.sequence_id,
                    **self._extra_failure_fields(candidate, context),
                })

        return ModuleVerdict(
            score=score,
            passed=not failures,
            classifications=classifications,
            details=details,
            failures=failures,
        )

    ###################### Parsing and Scoring ######################
    def _parse_classifications(self, evaluation_response: str, num_guidelines: int) -> Tuple[List[str], Dict]:
        classifications: List[str] = []
        details: Dict = {}

        try:
            current_guideline = -1
            for line in evaluation_response.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue

                if line.upper().startswith("GUIDELINE") and any(c.isdigit() for c in line):
                    current_guideline += 1
                    continue

                if line.upper().startswith("CLASSIFICATION:"):
                    text = line.split(":", 1)[1].strip().upper()
                    for label in self._match_order:      # worst label first
                        if label.upper() in text:
                            classifications.append(label)
                            break
                    else:
                        classifications.append(self._best_label)  # default
                elif line.upper().startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
                    if current_guideline >= 0:
                        details[f"guideline_{current_guideline}_reasoning"] = reasoning

            while len(classifications) < num_guidelines:
                classifications.append(self._best_label)
            for i in range(num_guidelines):
                details.setdefault(f"guideline_{i}_reasoning", "No specific reasoning provided")

        except Exception as e:
            logger.warning(f"Error parsing classifications: {e}")
            classifications = [self._best_label] * num_guidelines
            details = {f"guideline_{i}_reasoning": "Error in parsing, defaulted to "
                       + self._best_label for i in range(num_guidelines)}

        return classifications[:num_guidelines], details

    def _classifications_to_score(self, classifications: List[str]) -> float:
        if not classifications:
            return self._best_score  # nothing to violate
        return min(self.vocabulary.get(c, 0.0) for c in classifications)

    ###################### Extra Fields ######################
    def _extra_failure_fields(self, candidate, context: TurnContext) -> Dict:
        """Hook for presets to enrich failure records (e.g. conversation_flow)."""
        return {}