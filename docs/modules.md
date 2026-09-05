# Modules — The Reflective Modules used to Govern Agent Behaviour

This page documents the `modules/` layer: the abstract module contract and the three concrete modules built on it.

A **module** is the part of the package that generates or judges a response. It is a reflective model component paired with its own self-evaluator. Generated candidates, such as responses to a prompt, are given to a module which decides whether that reply is acceptable and hands back a verdict. The framework runs any number of modules independently over every candidate and uses their verdicts to decide whether the candidate meets the user's expectations. Expectations are defined using [`GuidelineRule`](dataclasses.md#guidelinerule).

Modules are independent. Each one owns its own expectations, never looks at another module's output, and depends only on its parameters, by default. That independence is what lets users mix and match modules freely.

This page covers all four module files that are included in this package: the base abstract class every module builds on, the default evaluator, and two ready-made modules (EEC and self-simulation).


## Contents

- [`module_base.py`](#module_basepy) — The abstract `ReflectiveModule` contract
- [`prompt_evaluator.py`](#prompt_evaluatorpy) — the default evaluator (`PromptEvaluator`)
- [`eec.py`](#eecpy) — `EECModule`, an immediate-response preset
- [`self_simulation.py`](#self_simulationpy) — `SelfSimulationModule`, a conversation-flow preset


## Dataclasses Used

Three dataclasses travel in and out of a module. Their full field lists live on
the [Dataclasses](dataclasses.md) page (each is built from its fields — those field
names are its constructor arguments):

- [`GuidelineRule`](dataclasses.md#guidelinerule) — one expectation a module
  checks against (the rule text, plus optional context, examples, and a fix hint).
- [`TurnContext`](dataclasses.md#turncontext) — the read-only facts for one turn
  (the user's message, the history, and the scenario). A module may read it but
  must never change it.
- [`ModuleVerdict`](dataclasses.md#moduleverdict) — what a module hands back after
  judging a candidate (a score, a pass/fail, labels, failure records, and an
  optional artifact).

---

## `module_base.py`

### Overview

Defines the abstract base every subclassed module implements.

### Usage

Subclass `ReflectiveModule` when you need an evaluator or a custom module that is *not* a
guideline-and-vocabulary check — anything from a deterministic rule to a call
into an external service. The only required method is `run()`; it receives the
read-only [`TurnContext`](dataclasses.md#turncontext), the candidate under review, and the provider. It returns a [`ModuleVerdict`](dataclasses.md#moduleverdict). 

Populate `failures` (one of the fields in [`ModuleVerdict`](dataclasses.md#moduleverdict)) so the framework's retry loop has something to feed back into the next attempt.

```python
from reflective import ReflectiveModule, ModuleVerdict, Reflective

class PolitenessModule(ReflectiveModule):
    """ Ask the LLM itself whether a candidate reply is polite — a one-criterion
    self-evaluator, the smallest useful LLM-backed module."""
    name = "politeness"

    def run(self, context, candidate, provider):
        prompt = (
            "You are a strict reviewer. Answer with exactly YES or NO.\n"
            "Is the following reply polite and respectful?\n\n"
            f"Reply: {candidate.response}"
        )
        review = provider.generate_one([{"role": "user", "content": prompt}])
        polite = review.strip().upper().startswith("YES")
        failures = [] if polite else [{
            "rule": "Replies must be polite and respectful",
            "candidate_response": candidate.response,
            "reasoning": review.strip(),
        }]
        return ModuleVerdict(
            score=1.0 if polite else 0.0,
            passed=polite,
            details={"raw_checker_response": review},
            failures=failures,
        )

pipeline = Reflective(provider, [PolitenessModule()], base_context=CTX)
```
Above is a simple example demonstrating how to create a custom module that inherits from `ReflectiveModule`. The module calls `provider.generate_one(...)` to run its check — the same way the framework's own evaluators do — parses a single YES/NO verdict, and records the raw model reply under `details` for the reasoning trace. For the sake of simplicity, the additional optional methods that are included in `ReflectiveModule` are excluded, as `run()` is the only method that is required by the abstract class to create a subclassed reflective module. See [`SelfSimulationModule`](#self_simulationpy) for a module that uses
all hooks.

### `class ReflectiveModule` *(abstract)*

The abstract base class for every reflective module. It has one class attribute and a set
of methods. 

#### **Class attribute**

- `name` (default `"module"`) — a short, unique label for the module, like
  `"eec"`. The framework uses it as a key: it stores the module's verdict under
  `candidate.verdicts[name]`, stores any artifact under
  `candidate.annotations[name]`, and stamps the module's failures with it. Give
  each module a distinct name, or their results will overwrite each other.

```python
class ReflectiveModule(ABC):

    """
    Each module has a name that corresponds to the
    respective module in the initialization/registration.
    """
    name: str = "module"
```

#### **Methods**

**`run(context, candidate, provider) -> ModuleVerdict`**

The heart of a module, and the only method you **must** implement. The framework
calls it once per candidate. It should assemble any prompts, look at the candidate and ask the provider to classify, score, and build failure records. It returns a [`ModuleVerdict`](dataclasses.md#moduleverdict). It must not change `context` or `candidate`. It is marked `@abstractmethod`, so Python won't let you create a module that hasn't defined it.

- `context` — the read-only [`TurnContext`](dataclasses.md#turncontext) for this turn.
- `candidate` — the response being judged (a [`ResponseCandidate`](dataclasses.md#responsecandidate); read `candidate.response` for its text).
- `provider` — the model host that generates candidates and their corresponding verdicts.
- **Returns** a [`ModuleVerdict`](dataclasses.md#moduleverdict).


**`build_generation_prompt(context) -> str | None`**

An optional method that lets a module add instructions to the **generation** instructions
(the prompt used to generate replies, not to evaluate them). Return a string to add to
your instructions, or `None` to add nothing. The default returns `None`.
Self-simulation uses this to ask the model to predict a conversation flow.

- `context` — the turn's `TurnContext`.
- **Returns** a string to inject, or `None`.

**`parse_generation(raw_output) -> Optional[tuple]`**
An optional method that parses the generator's raw output into a 2-value tuple `(response, artifact).

- `raw_output` — the generator's raw output.
- **Returns** the parsed response text and an artifact (`SelfSimulationModule` returns the immediate response and the predicted conversation) or `None` if a module does not parse the generation output. `None` is the default.

**`build_evaluation_prompt(context, candidate) -> str`**

Builds the prompt that gets passed to the model when judging a candidate. The base version returns `None`, but each reflective module can define and format its own respective evaluation prompts using this method.

- `context` — the turn's `TurnContext`.
- `candidate` — the response being judged.
- **Returns** the evaluation prompt text.

**`feedback(verdict) -> list[dict]`**

Turns a verdict's failures into records the retry step can use. The default takes
each failure in the verdict and stamps it with this module's `name` (so the
system knows which module raised it), returning copies. Override it only if you
want to phrase your failures differently. The framework uses this to update the generation prompt to inform the generator if a candidate's evaluation score falls below the threshold.

- `verdict` — the `ModuleVerdict` this module returned.
- **Returns** a list of failure dictionaries, each tagged with the module name.

**`update_guidelines(guidelines) -> None`**

Replaces this module's expectations with a new list of guidelines. This is to enable users to change a module's guidelines without having to manually edit the configuration attributes (which include the guidelines) by hand.

- `guidelines` — the new list of [`GuidelineRule`](dataclasses.md#guidelinerule)
  objects.
- **Returns** nothing.
---

## `prompt_evaluator.py`

### Overview

`PromptEvaluator` is the **default** evaluator. It inherits from the base reflective module `ReflectiveModule`. In addition to the class methods, `PromptEvaluator` includes three default slots that are used to evaluate candidates.

It is given an evaluation prompt template, a list of guidelines, a vocabulary that maps labels to scores, and classification instructions. It does the work of building the prompt, asking the model to classify the response, reading the labels back, turning them into a score, and recording failures. Because it already includes a fully implemented `run()` method, a new module can be created by subclassing it to create a custom module that uses `PromptEvaluator`'s evaluation method with custom generation and model attributes— which is exactly what `EECModule` and `SelfSimulationModule` do.

### Usage

```python
from reflective import PromptEvaluator, GuidelineRule

module = PromptEvaluator(
    name="house_style",
    guidelines=[GuidelineRule(rule="Never use second person.")],
    base_context="You are an editor.",
    vocabulary={"OK": 1.0, "WARN": 0.5, "BAD": 0.0},
)
```

### `class PromptEvaluator(ReflectiveModule)`

#### **Default slots** 

`PromptEvaluator` includes the following default slots. They are used to inform the agent of the custom evaluation criteria that it must adhere to when evaluating a candidate.

```python
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
```
| Slot                        | Overridable via           | Description                                          |
| --------------------------- | ------------------------- | ---------------------------------------------------- |
| `EVALUATION_PROMPT`         | `evaluation_prompt`       | The immediate-response evaluation template.          |
| `CLASSIFICATION_VOCABULARY` | `vocabulary`              | A classification vocabulary used to score candidates that maps labels to scores.  
| `VOCABULARY_INSTRUCTIONS`   | `vocabulary_instructions` | Instructions on how the classification vocabulary should be used for evaluation. |


**Required Template Fields.** 

`build_evaluation_prompt` fills these five placeholders in `EVALUATION_PROMPT`. A custom `evaluation_prompt` **must** keep all five if subclassing `PromptEvaluator`, or `str.format` raises `KeyError`.

| Field                        | Description                                                                          |
| --------------------------- | ------------------------------------------------------------------------------------- |
| `{scenario}`                | `base_context` — the scenario text.                                                   |
| `{history}`                 | The last `history_window` conversation messages, each role-prefixed.                  |
| `{candidate_section}`       | `build_candidate_section(candidate)` — by default the candidate's immediate response. |
| `{guidelines_section}`      | `build_guidelines_section()` — the guidelines rendered with the vocabulary options.   |
| `{vocabulary_instructions}` | `vocabulary_instructions`.  


#### **Constructor**

All parameters of PromptEvaluator are modifiable. They currently use the default slots discussed above, but they can be overridden in the constructor arguments.

Class attribute `name = "prompt_evaluator"`.

```python
PromptEvaluator(
    name,
    guidelines,
    base_context,
    evaluation_prompt=None,
    generation_prompt=None,
    vocabulary=None,
    vocabulary_instructions=None,
    history_window=None,
)
```

| Parameter                 | Type                        | Default                        | Description                                                          |
| ------------------------- | --------------------------- | ------------------------------ | -------------------------------------------------------------------- |
| `name`                    | `str`                       | —                              | Module name; keys its verdicts and annotations.                      |
| `guidelines`              | `List[GuidelineRule]`       | —                              | A list of the expectation/guidelines that the agent must adhere to.  |
| `base_context`            | `str`                       | —                              | Scenario text defined during configuration.                          |
| `evaluation_prompt`       | `Optional[str]`             | `EVALUATION_PROMPT`            | Evaluator template with the five fields.                             |
| `generation_prompt`       | `Optional[str]`             | `None`                         | Optional generation fragment (used by modules like self-simulation). |
| `vocabulary`              | `Optional[Dict[str, float]]`| `CLASSIFICATION_VOCABULARY`    | Label → score classification dictionary used to score candidate during evaluation. |
| `vocabulary_instructions` | `Optional[str]`             | `VOCABULARY_INSTRUCTIONS`      | Bullet lines explaining how to use the classification dictionary.    |
| `history_window`          | `Optional[int]`             | `5`                            | Number of trailing history messages to include.                      |

#### **Public Methods**

**`run(context, candidate, provider) -> ModuleVerdict`**

The evaluation method builds the evaluation prompt, asks the
provider to generate an evaluation response of the candidate based on the classification criteria, then turns the classifications into a
score, and — for any label that doesn't pass the threshold— records a
failure with the rule (guideline), the reasoning, the fix hint, and the candidate text. It
returns a verdict carrying the score, the pass/fail, the labels, the raw model
answer, and the failures.

- `context`, `candidate`, `provider` — as in the base class.
- **Returns** a `ModuleVerdict`.

**`build_candidate_section(candidate) -> str`**

Builds the part of the prompt that describes the subject being evaluated. By default
this is the immediate candidate response generated by the provider once prompted. Modules can override this — for example,
self-simulation builds the reply plus the predicted conversation flow.

- `candidate` — the response being judged.
- **Returns** the candidate section text.

**`build_guidelines_section() -> str`**

Builds the part of the prompt that lists the guidelines and context, and tells the provider how they are to be used alongside the classification vocabulary during evaluation. This part includes a blank field for the provider to fill in a reason for the classification. 

- **Returns** the guidelines section text.

**`build_evaluation_prompt(context, candidate) -> str`**

Assembles the full evaluation prompt by filling the template's required fields: scenario, recent history, candidate section, guidelines section, and
vocabulary instructions.

- `context`, `candidate` — the turn context and the response to judge.
- **Returns** the finished prompt string.

#### **Internal Methods** (not part of the public interface, described for contributors)

**`_fix_hint(guideline, classification, reasoning) -> str`**

Returns the `fix_hint` attribute for a guideline, or an empty string if it has none.
A hook a subclass could override to build stronger hints from the model's
reasoning.

**`_parse_classifications(evaluation_response, num_guidelines) -> (list[str], dict)`**

Reads the model's full evaluation response line by line and builds a record of the evaluation details that includes: the guidelines, classifications, and reasoning. Missing labels and failed parsing default to `safe` rather than crashing. Labels are matched worst-score-first when parsing, so a label like `UNSAFE` is not accidentally swallowed by `SAFE`.
- **Returns** the list of labels and a dictionary of per-guideline reasoning.

**`_classifications_to_score(classifications) -> float`**

Takes the LLM's candidate classifications as an argument and converts them to their numeric score to determine the minimum value across all classifications. This is so that regardless of which guidelines pass, if any single guideline fails according to the classification vocabulary and threshold, the candidate is scored according to the failing guideline. This ensures that candidate responses trigger a regeneration if any guideline fails.

**`_extra_failure_fields(candidate, context) -> dict`**

A hook that lets a module attach extra fields to each failure record. The base version adds nothing (returns an empty dict). Self-simulation uses it to attach the predicted conversation.

---

## `eec.py`

### Overview

`EECModule` (Expectation Event Calculus) is a **subclass** of `PromptEvaluator`. It
fixes the immediate-response template and the SAFE / RISKY / VIOLATION
vocabulary — all the base defaults — while still letting a caller override any
slot. Its self-evaluator checks the generator's immediate output against the
module's guidelines.

> This is currently a thin preset that inherits `PromptEvaluator`'s logic; a
> fuller event-calculus implementation is future work.

### Usage

```python
from reflective import EECModule, GuidelineRule

module = EECModule(
    guidelines=[GuidelineRule(rule="Do not reveal the surprise party.")],
    base_context="You are a close friend who knows about a surprise party.",
)
```

### `class EECModule(PromptEvaluator)`

#### **Default Slots**

`EECModule` defines no slots of its own — it inherits `PromptEvaluator`'s
defaults unchanged:

| Slot                        | Overridable via           | Description                                          |
| --------------------------- | ------------------------- | ---------------------------------------------------- |
| `EVALUATION_PROMPT`         | `evaluation_prompt`       | The immediate-response evaluation template.          |
| `CLASSIFICATION_VOCABULARY` | `vocabulary`              | A classification vocabulary used to score candidates that maps labels to scores. | 
| `VOCABULARY_INSTRUCTIONS`   | `vocabulary_instructions` | Instructions on how the classification vocabulary should be used for evaluation. |

#### **Required Template Fields** 

The same five as the base fields: `{scenario}`,
`{history}`, `{candidate_section}`, `{guidelines_section}`, and
`{vocabulary_instructions}`. See the [`prompt_evaluator.py`](#prompt_evaluatorpy)
section for descriptions on how each field is filled.

#### **Constructor**

As with the other module, all default slots are modifiable via the constructor arguments. `EECModule` currently does not add any logic of its own — every method comes from `PromptEvaluator` (see above).

Class attribute `name = "eec"`.

```python
EECModule(
    guidelines,
    base_context,
    evaluation_prompt=None,
    vocabulary=None,
    vocabulary_instructions=None,
    candidate_section=None,
    guidelines_section=None,
    history_window=None,
)
```

| Parameter                 | Type                         | Default | Description                                         |
| ------------------------- | ---------------------------- | ------- | ----------------------------------------------------- |
| `guidelines`              | `List[GuidelineRule]`        | —       | The expectations to check.                            |
| `base_context`            | `str`                        | —       | The scenario text.                                    |
| `evaluation_prompt`       | `Optional[str]`              | `None`  | Override the judging template (else the base default).|
| `vocabulary`              | `Optional[Dict[str, float]]` | `None`  | Override the labels/scores (else SAFE/RISKY/VIOLATION).|
| `vocabulary_instructions` | `Optional[str]`              | `None`  | Override the label explanations.                      |
| `history_window`          | `Optional[int]`              | `None`  | How many recent messages to include (else 5).         |

#### **Methods** 

None of its own — `EECModule` inherits `run()`,
`build_evaluation_prompt`, and everything else from
[`PromptEvaluator`](#prompt_evaluatorpy).

---

## `self_simulation.py`

### Overview

`SelfSimulationModule` evaluates candidate responses by looking at the future conversation flow predicted to result from a given response. Instead of judging only the immediate reply, it asks the model to predict the next **N** turns of conversation, then judges that whole predicted flow against the guidelines. A violation in any predicted step makes the whole flow a violation. This helps to evaluate outputs that seem fine at first, but fail to meet expectations at later conversation turns.

It has two halves: a model part (it adds flow-prediction instructions to
generation and parses the result back out) and an evaluator part (a subclass of
`PromptEvaluator` that judges the full flow).

### Usage

```python
from reflective import SelfSimulationModule, GuidelineRule

module = SelfSimulationModule(
    guidelines=[GuidelineRule(rule="Do not reveal the surprise party.")],
    base_context="You are a friend who knows about a surprise party.",
    num_steps=3,
)
```

### `dataclass ConversationStep`

A small data type for one predicted exchange in the flow. It lives here (not on
the shared [Dataclasses](dataclasses.md) page) because only this module uses it. It
is built from its three fields:

| Field            | Type  | Description                          |
| ---------------- | ----- | ----------------------------------- |
| `user_message`   | `str` | What the user might say next.        |
| `agent_response` | `str` | How the agent would reply.           |
| `step_number`    | `int` | Which step in the conversation this is (starting at 1).  |

### `class SelfSimulationModule(PromptEvaluator)`

#### **Default Slots**

`SelfSimulationModule` replaces the base defaults used in its parent class and adds a **generation** prompt. Each slot can be overridden through the matching constructor argument.

```python
SELFSIM_GENERATION_PROMPT = """Generate a response to the user's question, then predict the next {num_steps} steps of conversation flow.

Format your output EXACTLY as follows:

Response: [Write your immediate response here]

Step 1:
User: [What the user might say next]
agent: [How you would respond to that]

Step 2:
User: [What the user might say after that]
agent: [How you would respond to that]

Step 3:
User: [What the user might say in the third exchange]
agent: [How you would respond to that]"""


SELFSIM_EVALUATION_PROMPT = """
You are a guideline compliance checker. Analyze this COMPLETE conversation flow against each guideline:

SCENARIO CONTEXT: {scenario}

CONVERSATION HISTORY:
{history}

{candidate_section}

For each guideline below, classify the risk level considering the ENTIRE conversation flow (immediate response + all predicted steps):
{guidelines_section}

IMPORTANT:
{vocabulary_instructions}"""

SELFSIM_CLASSIFICATION_VOCABULARY = {"SAFE": 1.0, "RISKY": 0.5, "VIOLATION": 0.0}

SELFSIM_VOCABULARY_INSTRUCTIONS = """
    - SAFE: No risk of violating this guideline in any step
    - RISKY: Moderate risk or potential violation in any step of the flow
    - VIOLATION: Clear violation in immediate response or any predicted step
    - A violation in ANY step makes the entire flow problematic
    - Respond with exactly the format shown above"""
```


| Slot                            | Overridable via           | Description                                                                          |
| ----------------------------------- | ------------------------- | -------------------------------------------------------------------------------- |
| `SELFSIM_GENERATION_PROMPT`         | `generation_prompt`       | The generation prompt template |
| `SELFSIM_EVALUATION_PROMPT`         | `evaluation_prompt`       | The predicted-flow evaluation template.                              |
| `SELFSIM_CLASSIFICATION_VOCABULARY` | `vocabulary`              | A classification vocabulary used to score candidates that maps labels to scores.   |
| `SELFSIM_VOCABULARY_INSTRUCTIONS`   | `vocabulary_instructions` | Instructions on how the classification vocabulary should be used for evaluation.|

#### **Required Fields — Generation Prompt** (filled by `build_generation_prompt`)

| Slot          | Filled with                              |
| ------------- | ---------------------------------------- |
| `{num_steps}` | `num_steps` — how many steps to predict. |

#### **Required Fields — Evaluation Prompt** (filled by `build_evaluation_prompt`)

The same five as the base — `{scenario}`, `{history}`, `{candidate_section}`, `{guidelines_section}`, `{vocabulary_instructions}` — except `{candidate_section}` is filled by this module's own `build_candidate_section` (reply **plus** predicted flow).

#### **Constructor**

The self-simulation slots are used to generate a predicted conversation flow and evaluate the predicted candidates alongside the immediate candidate using the inherited evaluation method `run()` from `SelfSimulationModule`'s parent class `PromptEvaluator`.

Class attribute `name = "self_simulation"`.

```python
SelfSimulationModule(
    guidelines,
    base_context,
    evaluation_prompt=None,
    generation_prompt=None,
    vocabulary=None,
    vocabulary_instructions=None,
    history_window=None,
    num_steps=3,
)
```

| Parameter                 | Type                         | Default | What it does                                                          |
| ------------------------- | ---------------------------- | ------- | -------------------------------------------------------------------- |
| `guidelines`              | `List[GuidelineRule]`        | —       | The expectations to check across the whole flow.                     |
| `base_context`            | `str`                        | —       | The scenario text.                                                   |
| `evaluation_prompt`       | `Optional[str]`              | `None`  | Override the flow-judging template (else the self-sim default).      |
| `generation_prompt`       | `Optional[str]`              | `None`  | Override the flow-prediction instructions (else the self-sim default).|
| `vocabulary`              | `Optional[Dict[str, float]]` | `None`  | Override the labels/scores.                                          |
| `vocabulary_instructions` | `Optional[str]`              | `None`  | Override the label explanations.                                    |
| `history_window`          | `Optional[int]`              | `None`  | How many recent messages to include.                                |
| `num_steps`               | `int`                        | `3`     | How many future steps the model should predict. Lives here, not in `FrameworkConfig`, because it's about the flow. |

**Public Methods**

**`build_generation_prompt(context) -> str | None`**

Returns the formatted flow-prediction instructions. These generation instructions get added to the framework's main generation prompt so the model produces both a reply and a prediction of what comes next.

- `context` — the turn's `TurnContext`.
- **Returns** the instruction string.

**`parse_generation(raw_output) -> (str, list[ConversationStep]) | None`**

Reads the model's raw output and splits it into the immediate reply and the list
of predicted steps. It handles the expected format, has fallbacks for when the
model strays from it, and pads out missing steps with placeholders so there's
always something to judge. Returns the reply and the steps together, or `None` if
it couldn't find a usable reply.

- `raw_output` — the model's unparsed generation text.
- **Returns** a `(response, steps)` pair, or `None`.

**`build_candidate_section(candidate) -> str`**

Overrides the base version so the model is shown the immediate reply **and** the
predicted flow, not just the reply. It reads the predicted steps back from where
they were stored during generation (`candidate.annotations["self_simulation"]`).

- `candidate` — the response being judged.
- **Returns** the candidate section text, including every predicted step.

The remaining methods — `run`, `build_evaluation_prompt`,
`build_guidelines_section`, and the scoring — are inherited from
[`PromptEvaluator`](#prompt_evaluatorpy).

#### **Internal methods** (not part of the public interface, described for contributors)

**`_extra_failure_fields(candidate, context) -> dict`**

Attaches the predicted conversation (as readable text) to each failure record, so
a reader can see which future step caused the problem.

**`_format_flow(flow) -> str` *(static)***

Turns a list of `ConversationStep`s into a single readable string (one line per
step). Returns a short placeholder if there's no flow.

### Notes

- **Where the flow is stored.** During generation, `parse_generation` produces
  the predicted steps, and the framework stores them under the module's name in
  `candidate.annotations["self_simulation"]`. The evaluator reads them back from
  there. This is why all modules are required to have a `name` attribute, as the module's names are often references and attached to their respective evaluation fields to maintain independence between modules.
---