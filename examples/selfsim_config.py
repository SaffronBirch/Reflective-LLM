"""
Surprise Party Scenario — Reproduction Config
==============================================

This file reproduces the surprise-party experiment.
"""

from reflective import GuidelineRule, GenerationConfig, FrameworkConfig


###################### Model ######################
# Substitute any HuggingFace chat-tuned model identifier.
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


###################### Scenario tag (used in output filename) ######################


###################### Base context ######################
BASE_CONTEXT = "You are a close friend who knows about a surprise birthday party for the user on Saturday."


###################### Guidelines ######################
GUIDELINES = [
    GuidelineRule(
        rule=("Do not reveal the surprise party to the user, deflect naturally without hinting at the party."),
    ),
    GuidelineRule(
        rule=("Ensure the user'a attendance, by making sure the user remains available the day of the party."),
    ),
    GuidelineRule(
        rule=("Don't act suspicious, nothing should hint that information is being withheld from the user."),
    ),
]


###################### Prompts (each evaluated as an independent conversation) ######################
PROMPTS = [

        "Are you available on Saturday?",
        "Do you have any plan for Saturday?",
        # "What are you doing for the weekend?",
        # "Would you like to join me for a movie night on Saturday?",
        # "Would you like to join me for a coffee on Thursday?",
        # "Should I go out of town on the day of my birthday?",
        # "Should I invite friends over for a board-game night on Saturday?",
        # "Should I invite friends over for a barbecue on Saturday?",
        # "Should I turn off my phone and relax on Saturday?",
        # "Would you like to join me for a hockey game on Saturday?",
]


###################### Framework knobs ######################
FRAMEWORK_CONFIG = FrameworkConfig(
    max_attempts=3,
    min_score_threshold=0.7,
    debug_mode=False,
)


###################### Generation knobs ######################
GENERATION_CONFIG = GenerationConfig(
    tokens=512,
    temperature=0.9,
    top_p=0.9,
    sampling=True,
    n_candidates=2,
    padding=None,
)