"""
Reflective — a self-evaluation loop for LLM-based systems.

    from reflective import Reflective, EECModule, SelfSimulationModule, GenerationConfig
    from reflective import HFModel            # imports torch lazily, on access

    modules = [EECModule(guidelines, base_context=CTX),
               SelfSimulationModule(guidelines, base_context=CTX, num_steps=3)]
    pipeline = Reflective(HFModel(model_name="..."), modules, base_context=CTX)
    reply = pipeline("Are you free Saturday?")
"""

import logging

# Public API — all torch-free, so `import reflective` stays lightweight.
from .architecture.framework import AgenticFramework
from .architecture.pipeline import Reflective
from .architecture.capture import capture_logging
from .architecture.dataclasses import (
    FrameworkConfig, 
    GenerationConfig, 
    TurnResult, 
    ResponseCandidate, 
    TurnContext, 
    GuidelineRule, 
    ModuleVerdict) 

from .modules.module_base import ReflectiveModule
from .modules.prompt_evaluator import PromptEvaluator
from .modules.eec import EECModule
from .modules.self_simulation import SelfSimulationModule

from .providers.provider_base import Provider


# Quiet by default: a no-op handler means importing the library adds no logging
# output to a host application. The host opts in with its own handler.
logging.getLogger("reflective").addHandler(logging.NullHandler())

__version__ = "0.1.0"

__all__ = [
    "Reflective", "AgenticFramework", "FrameworkConfig", "TurnResult",
    "GuidelineRule", "ReflectiveModule", "ModuleVerdict", "TurnContext",
    "PromptEvaluator", "EECModule", "SelfSimulationModule", "Provider", 
    "GenerationConfig", "HFModel", "ResponseCandidate", "capture_logging",
]

def __getattr__(name: str):
    # Lazy: `from reflective import HFModel` works, but torch is imported only
    # when HFModel is actually accessed, not on `import reflective`.
    if name == "HFModel":
        from .providers.hugging_face import HFModel
        return HFModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")