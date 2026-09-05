"""
To run, use the following command in the root directory (reflective-llm)
PYTHONPATH=. python examples/run_selfsim_evaluation.py
"""
###################### Imports ######################
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("reflective").setLevel(logging.INFO) 

from reflective import Reflective, EECModule, SelfSimulationModule, HFModel

from examples.selfsim_config import (
    MODEL_NAME, SCENARIO_NAME, BASE_CONTEXT, GUIDELINES, PROMPTS,
    FRAMEWORK_CONFIG, GENERATION_CONFIG,
)

###################### SelfSimulationModule Instantiation ######################
pipeline = Reflective(
    provider=HFModel(model_name=MODEL_NAME, generation_config=GENERATION_CONFIG),
    modules=[
        EECModule(list(GUIDELINES), base_context=BASE_CONTEXT),
        SelfSimulationModule(list(GUIDELINES), base_context=BASE_CONTEXT, num_steps=3),
    ],
    base_context=BASE_CONTEXT,
    framework_config=FRAMEWORK_CONFIG,
    scenario_name=SCENARIO_NAME,
)

# Batch run through the single entry point: each prompt is an independent
# conversation, and one aggregated results JSON + .log is written for the whole
# batch (what a standalone batch runner used to produce). Returns TurnResults.
results = pipeline.run_batch(PROMPTS)