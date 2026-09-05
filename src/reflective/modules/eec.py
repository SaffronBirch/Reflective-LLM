'''
EEC module: expectations defined at the module level, with a self-evaluator
that checks the generator's immediate output against them.

A preset of PromptEvaluator: it fixes the immediate-response template and the
SAFE / RISKY / VIOLATION vocabulary (all the base defaults), while still
letting a caller override any slot.
'''

###################### Imports ######################
from typing import Callable, Dict, List, Optional

from ..architecture.dataclasses import GuidelineRule
from .prompt_evaluator import PromptEvaluator

###################### EEC ######################
class EECModule(PromptEvaluator):

    name = "eec"

    def __init__(
        self,
        guidelines: List[GuidelineRule],
        base_context: str,
        evaluation_prompt: Optional[str] = None,
        vocabulary: Optional[Dict[str, float]] = None,
        vocabulary_instructions: Optional[str] = None,
        history_window: Optional[int] = None,
    ):
        super().__init__(
            name="eec",
            guidelines=guidelines,
            base_context=base_context,
            evaluation_prompt=evaluation_prompt,
            vocabulary=vocabulary,
            vocabulary_instructions=vocabulary_instructions,
            history_window=history_window,
        )