###################### Imports ######################
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from ..architecture.dataclasses import GuidelineRule, TurnContext, ModuleVerdict

###################### Abstract Class ######################
class ReflectiveModule(ABC):
    """
    A reflective module per the architecture in Salmani & Lewis: a model
    component paired with its own self-evaluator. Internally a module may
    organize itself that way (see EECModule, SelfSimulationModule), but the
    framework contract is only ``run``.

    Modules are independent: they own their expectations, they never read
    another module's output, and ``run`` must depend only on its arguments
    and the module's own configuration. This is what makes modules swappable
    and the framework's fan-out safe to parallelize later.
    """

    name: str = "module"
###################### Formatting and parsing for candidate generation ######################
    def build_generation_prompt(self, context: TurnContext) -> Optional[str]:
        """
        Optional prompt fragment this module wants injected into the
        generation prompt (e.g. self-simulation's flow-prediction format).
        Return None if this module adds nothing to generation (the default).
        """
        return None

    def parse_generation(self, raw_output: str) -> Optional[tuple]:
        """
        Optionally parse the generator's raw output into
        ``(response_text, artifact)``. The orchestrator stores the artifact
        under ``candidate.annotations[module.name]``. Return None if this
        module does not parse generation output (the default).
        """
        return None

###################### Formatting for evaluation genereration  ######################
    def build_evaluation_prompt(self, context: TurnContext, candidate) -> str:
        return None
        
###################### Evaluation execution  ######################
    @abstractmethod
    def run(self, context: TurnContext, candidate, provider) -> ModuleVerdict:
        """Evaluate one candidate. Must not mutate ctx or candidate."""
        pass

###################### Prompt revision for failed candidates  ######################
    def feedback(self, verdict: ModuleVerdict) -> List[Dict]:
        """
        How this module phrases a verdict's failures for the adapter's prompt
        revision. Default: pass the failures through, stamped with this module's
        name so the adapter can scope them. Returns copies — the adapter merges
        records in place and must not reach back into the verdict.
        """
        return [{**f, "module": self.name} for f in verdict.failures]

# ###################### Updating guidelines  ######################
    def update_guidelines(self, guidelines: List[GuidelineRule]) -> None:
        """
        Replace this module's expectations wholesale. The front door for any
        later mutation (e.g. a dynamic EEC implementation); callers should
        use this rather than assigning module attributes directly.
        """
        self.guidelines = guidelines