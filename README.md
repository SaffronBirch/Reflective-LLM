# Reflective-LLM

**A modular reflective framework for large language model-based systems.**

This framework situates a language model within a closed reflective loop. Rather than returning the model's initial, unvalidated output, the system treats each generation as a set of *candidate behavioral outputs*, subjects them to *self-evaluation* against a set of explicit expectations, and returns a candidate only once it satisfies those expectations. If no candidate does, the accumulated evaluation feedback is used to revise the prompt and regenerate, up to a bounded number of attempts, after which the highest-scoring candidate is returned. The entire mechanism is introduced at a single call site.

The framework is task-agnostic. It applies to any setting in which a language model produces outputs whose alignment with human expectations, values, or social norms is consequential — including safety, social appropriateness, tone, factual reliability, persona consistency, and policy compliance. Two reflective modules are provided as reference implementations to demonstrate how LLMs can validate their own outputs when equipped with self-reflection capabilities. Additional modules can be defined to customize the usage capabilities of this framework. All modules are designed to be independent and swappable to meet the needs of any application where LLMs generate textual responses.

Both candidate generation and evaluation are governed by *slots* (see [Definitions](#definitions--key-concepts)), enabling the framework to be adapted to a given application without modifying its internal parameters directly.

The design is a working implementation of the architecture proposed by **Salmani & Lewis, *A Reflective Architecture for LLM-based Systems*** (IEEE ACSOS-C 2025). This document maps the implementation to the described architecture and documents its installation and use.

---

## Table of contents

- [Overview](#overview)
- [How it works](#how-it-works)
- [Definitions & key concepts](#definitions--key-concepts)
- [How it maps to the paper](#how-it-maps-to-the-paper)
- [The three layers](#the-three-layers)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [The pipeline, start to finish](#the-pipeline-start-to-finish)
- [Creating a custom module](#creating-a-custom-module)
- [Custom vocabularies](#custom-vocabularies)
- [Writing a custom provider](#writing-a-custom-provider)
- [Documentation](#documentation)
- [API reference](#api-reference)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Citation](#citation)

---

## Overview

A conventional language model produces its response in a single forward pass, with no opportunity to monitor or correct its own behaviour; what appears to be reflection is a property of the generated text rather than a genuine evaluative process. The paper argues that reliable, socially aware behaviour instead requires proper reflection. The system should validate a candidate output against its expectations (as defined by the user), anticipate the consequences of committing to it, and revise before returning it. This package realizes that argument as a compact, modular loop that can be placed around any provider. Because the expectations are supplied by the user, this package can also be applied to any task in which its objective can be described as a set of guidelines.

```python
from reflective import Reflective, EECModule, SelfSimulationModule, GuidelineRule, HFModel

CTX = "You are a close friend who knows about a surprise party for the user on Saturday."
guidelines = [
    GuidelineRule(rule="Do not reveal the surprise party; deflect naturally."),
    GuidelineRule(rule="Keep the user available on Saturday."),
    GuidelineRule(rule="Do not act suspicious."),
]

pipeline = Reflective(
    provider=HFModel(model_name="mistralai/Mistral-7B-Instruct-v0.3"),
    modules=[
        EECModule(list(guidelines), base_context=CTX),
        SelfSimulationModule(list(guidelines), base_context=CTX, num_steps=3),
    ],
    base_context=CTX,
)

reply = pipeline("Are you free Saturday?")   # a validated response, not the raw output
```

The surprise-party scenario above is the social-normative example used in the
paper's evaluation. The specific scenario, base context, and guidelines can be changed to match a different objective while leaving the mechanical structure unchanged. For example, enforcing medical disclaimers, mitigating socially biased outputs, maintaining a game character's persona, or any other set of expectations.

---


## How it works

Each turn runs as a closed reflective loop:

1. **Generate** multiple candidate behavioral outputs from the provider.
2. **Self-evaluate** each candidate by submitting it to every enabled reflective module. Each module returns a verdict.
3. **Reconcile** the module verdicts into a single score. By default the
   reconciliation selects the score that is the minimum across modules, so a
   single objection is sufficient to reject a candidate.
4. **Accept or revise.** A candidate whose score meets the threshold is returned;
   otherwise the modules' failures are compiled into corrective feedback, appended
   to the next attempt's prompt, and the loop repeats up to `max_attempts`.
5. **Fall back** to the highest-scoring candidate observed if no attempt satisfies
   the threshold.

The loop is exposed through a single object, `Reflective`, which serves as the
package's sole entry point.

---

## The Three Layers

An application selects a **provider**, assembles a set of **modules**, and supplies
both to the **architecture** layer's pipeline.

- **Providers** — model backends behind a common interface, so that substituting
  one for another requires minimal effort. `HFModel` is
  provided for HuggingFace models. See [docs/providers.md](docs/providers.md).
- **Modules** — self-evaluators that assess candidates against expectations.
  They are independent and swappable.
  See [docs/modules.md](docs/modules.md).
- **Architecture** — the reflective loop (`AgenticFramework`), the pipeline that
  hosts it (`Reflective`), batch execution, and log capture. See
  [docs/framework.md](docs/framework.md) and [docs/pipeline.md](docs/pipeline.md).

---


## Key Concepts and Definitions

The following terms are used throughout the project. They are defined in the context of this framework and aligned with the terminology used in the source paper.

### Key Components

- **`Reflective`** — the pipeline and the single entry point. It is constructed
  with a provider and a set of modules, and is invoked as a callable.
- **`Provider`** — the behavioral generator. It produces candidate outputs and,
  when a module requests it, performs the model call used for evaluation. `HFModel`
  is the built-in HuggingFace provider; alternative providers may be implemented
  for any model or service.
- **`ReflectiveModule`** — a self-evaluator that assesses
  candidate outputs against a set of expectations. Two modules are provided (`EECModule`, `SelfSimulationModule`),
  both derived from the reusable `PromptEvaluator` module, which serves as the default evaluator; applications can configure these or
  define their own.
- **`AgenticFramework`** — the reflective loop itself (generate → self-evaluate →
  reconcile → adapt → retry). `Reflective` wraps it.

### Customizable Slots

Slots are the mechanism through which both the modules and the framework can be customized to adapt to any given application without modifying their internals:

- **Module slots** govern *how a module evaluates.* In `PromptEvaluator` (and the
  modules derived from it) the slots comprise the evaluation template
  (`evaluation_prompt`), the classification label set and associated scores (`vocabulary`), the
  label definitions, descriptions and usage instructions (`vocabulary_instructions`), any generation instructions
  (`generation_prompt`), and the amount of history included (`history_window`).
  Replacing the vocabulary, for instance, from SAFE/RISKY/VIOLATION to
  PASS/WARN/FAIL, or substituting the template causes the same module to enforce a
  different standard; see [Custom vocabularies](#custom-vocabularies).
- **Framework slots** govern *how candidates are generated.* The generation prompt
  is assembled from named slots — `feedback` (the corrective section derived from
  prior failures), `attempt_guidance` (progressively stronger guidance on subsequent
  attempts), `instructions`, and `footer`. All are overridable through the `slots`
  argument to `Reflective`.

This is what makes `reflective` a general framework rather than a fixed tool: the
loop, the revision cycle, and the scoring remain constant, while slots direct the
framework at whatever expectations and evaluation method a given application defines.

---

## How it Maps to the Paper

The paper's architecture is a left-to-right pipeline: a **sensor** updates a world
model; a **behavioral LLM generator** proposes candidate outputs; two validation
paths (**EEC** and **self-simulation**, each with an associated **self-evaluator**)
assess them; a **reconciler** combines the assessments; a **(prompt) adapter**
returns corrections to the generator; and an **integrator** consolidates the final
decision. This package implements that control flow as interchangeable components.

| Paper | Package component | Role |
|---|---|---|
| Sensor → Updated State of the World | `TurnContext` | Holds the user input, history, and scenario context for one turn (read-only) | 
| Expectations and guidelines | `base_context` + `GuidelineRule`s | The expectations the output must satisfy | 
| Behavioral (LLM) Generator | `Provider` / `HFModel` | Produces *N* candidate completions per turn | 
| EEC Model + self-evaluator | `EECModule` | Assesses the immediate response against the expectations |
| Self-simulation + self-evaluator | `SelfSimulationModule` | Predicts the next *N* steps and assesses the projected trajectory | 
| Reconciler | `AgenticFramework.reconcile` | Combines module verdicts; accepts or triggers regeneration | 
| (Prompt) Adapter | `feedback()` + feedback slot + revision loop | Converts failures into prompt revisions for the next attempt (≤ `max_attempts`) |
| Integrator | Final selection + `TurnResult` | Consolidates the accepted (or fallback) candidate | 
| Actuator → Environment | *(returning the reply)* | — | 

> **Note about EEC Model:** The package implements the *control flow and modular structure* of the architecture, together with a proof-of-concept realization of the internal components. The EEC model implementation follows the paper's scope, which uses this framework as a way to demonstrate the architecture structure's capabilities. Therefore, the full EEC model is not yet implemented, but rather treats expectations as author-supplied rules and assesses compliance via an LLM prompt. The full implementation is left to future work. A formal expectation-generation mechanism can be integrated behind `ReflectiveModule.update_guidelines()`.

---

## Repository structure

```
reflective-llm/
├── docs/                           # documentation (see the Documentation section)
│   ├── providers.md
│   ├── modules.md
│   ├── framework.md
│   ├── pipeline.md
│   ├── dataclasses.md
├── examples/
│   ├── selfsim_config.py           # scenario settings
│   └── run_selfsim_evaluation.py
├── src/
│   └── reflective/                 # the import package
│       ├── architecture/
│       │   ├── __init__.py
│       │   ├── dataclasses.py      # shared data types
│       │   ├── framework.py        # AgenticFramework (the loop)
│       │   ├── pipeline.py         # Reflective (the entry point)
│       │   └── capture.py          # internal log capture
│       ├── modules/
│       │   ├── __init__.py
│       │   ├── module_base.py      # ReflectiveModule (base class)
│       │   ├── prompt_evaluator.py # PromptEvaluator
│       │   ├── eec.py              # EECModule
│       │   └── self_simulation.py  # SelfSimulationModule
│       └── providers/
│           ├── __init__.py
│           ├── provider_base.py    # Provider, GenerationConfig
│           └── hugging_face.py     # HFModel
├── __init__.py                             
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

Execution also produces a `reflective_results/` directory (saved JSON and `.log`
files) by default; it is generated at runtime and is not part of the source.

---

## Installation

Requires **Python 3.9+**.

To install:

```bash
git clone https://github.com/your-org/reflective-llm
cd reflective-llm
pip install -e .            # core only
pip install -e ".[hf]"      # with the built-in HuggingFace provider (torch, transformers, accelerate)
```

> **Note on dependencies:** The core package (framework, pipeline, and modules) depends only on the standard library. `torch` and `transformers` are required solely by `HFModel` and are imported only when that provider is imported to keep the package lightweight.

---

## The pipeline, start to finish

Invoking `pipeline(...)` transfers control to
`AgenticFramework.main_chat_handler`. The stages, in order:

**1. Assemble the world state:** The user input, history, and `base_context` are
assembled into a read-only `TurnContext` — the paper's *Updated State of the
World*.

**2. Enter the attempt loop:** Repeats up to `FrameworkConfig.max_attempts`.

**3. Generate candidates:** A generation prompt is assembled from named *slots* —
scenario context, history, accumulated feedback from previous attempts, per-attempt
guidance, and instructions. Modules may contribute to generation via
`build_generation_prompt()`: `SelfSimulationModule` inserts its "write a response,
then predict the next N turns" format here. The `Provider` then returns *N*
candidate strings — the paper's *behavioral generator* producing candidate outputs.

**4. Parse candidates:** Each candidate is offered to any module that parses
generation output via `parse_generation()`. `SelfSimulationModule` separates its
output into the immediate response and the predicted trajectory, storing the latter
on the candidate under the module's name.

**5. Self-evaluate:** For each candidate, every module's `run()`
executes **independently** and returns a `ModuleVerdict`. These are the paper's two
parallel self-evaluators: `EECModule` assesses the immediate response, while
`SelfSimulationModule` assesses the response together with its predicted
trajectory, identifying responses that are acceptable in isolation but may lead to
undesirable responses at later stages.

**6. Reconcile:** The framework combines each candidate's verdicts —
the candidate's score is the minimum across modules. A candidate that meets
`min_score_threshold` is accepted, and the loop terminates. This is the paper's
*reconciler*.

**7. Adapt and revise:** If no candidate is accepted, each module renders its
failures via `feedback()`; these records are placed in the feedback slot, and the
next attempt regenerates with that corrective guidance incorporated into the
prompt. This is the paper's *(prompt) adapter* closing the loop.

**8. Integrate:** If the attempts are exhausted without any candidate meeting the
threshold, the highest-scoring candidate is selected so that the caller always
receives a response (marked `fallback=True`). This is the paper's *integrator*.

**9. Return a `TurnResult`:** The returned response together with the full
reasoning trace, all candidates, the winning score, and the accumulated feedback.

---

## Quickstart

### Using the Provided Modules

Two modules are included: `EECModule` (which assesses the immediate response) and
`SelfSimulationModule` (which assesses the immediate response together with a
predicted continuation of the conversation). Construct them, supply them to
`Reflective`, and invoke the pipeline in place of the model.

```python
from reflective import Reflective, EECModule, SelfSimulationModule, GuidelineRule, HFModel, FrameworkConfig

context = "You are a close friend who knows about a surprise party for the user on Saturday."

guidelines = [
    GuidelineRule(rule="Do not reveal the surprise party; deflect naturally."),
    GuidelineRule(rule="Keep the user available on Saturday."),
    GuidelineRule(rule="Do not act suspicious."),
]

pipeline = Reflective(
    provider=HFModel(model_name="mistralai/Mistral-7B-Instruct-v0.3"),
    modules=[
        # Each module owns its guideline list; construct one per module.
        EECModule(list(guidelines), base_context=context),
        SelfSimulationModule(list(guidelines), base_context=context, num_steps=3),
    ],
    base_context=context,
    framework_config=FrameworkConfig(max_attempts=3, min_score_threshold=0.7),
)

# Invoke with a string...
print(pipeline("Are you free Saturday?"))

# ...or with a full chat history (the final user message is the turn's input):
messages = [
    {"role": "user", "content": "Hey!"},
    {"role": "assistant", "content": "Hi! What's up?"},
    {"role": "user", "content": "Do you have plans Saturday?"},
]
print(pipeline(messages))
```

> **Note on guideline ownership.** A module owns the guideline list with which it is constructed. If two modules are to share the same expectations, pass them to both modules `list(guidelines)`.

### Obtaining the Full Result

Passing `full_result=True` returns a `TurnResult` in place of the response string:

```python
result = pipeline("Are you free Saturday?", full_result=True)
print(result.final_response)   # the response
print(result.success)          # whether a candidate met the threshold
print(result.winning_score)    # its reconciled score
print(result.attempts_used)    # the number of generate/evaluate rounds
print(result.fallback)         # True if no candidate passed and the fallback was returned
```

### Batch Execution

The pipeline can be used to run an evaluation on a batch of prompts. Each prompt is processed as an independent conversation, and a single aggregated
results file is written for the batch.

```python
results = pipeline.run_batch(["Are you free Saturday?", "Any plans this weekend?"])
```
---

## Create a Custom Module

Most custom modules are subclasses of `PromptEvaluator`, the default evaluator that provides prompt
assembly, LLM-based classification, scoring, and failure records. The subclass
supplies a name, guidelines, and, optionally, its own slots. A module is used by constructing it
and supplying it to `Reflective`.

```python
from reflective import PromptEvaluator

class ToxicCommentModule(PromptEvaluator):
    name = "toxic_comment"
    def __init__(self, guidelines, base_context):
        super().__init__(name="toxic_comment", guidelines=guidelines, base_context=base_context)
```

Used directly:

```python
from reflective import Reflective, GuidelineRule, HFModel

CTX = "You are a toxic comment detector."
g = [GuidelineRule(rule="Flag any comment that contains harsh or innapropriate content.")]

pipeline = Reflective(HFModel(model_name="..."), [BrandVoiceModule(g, base_context=CTX)], base_context=CTX)
```

### A module that contains a Generative Component

A module that both contributes to the generation prompt and parses structured
output from it (as self-simulation does) implements two methods:

- `build_generation_prompt(context) -> str | None` — a base `ReflectiveModule`
  hook; returns text to append to the generation prompt.
- `parse_generation(raw_output) -> (response, artifact) | None` — separates the raw
  output into the immediate response and an additional artifact. The artifact is stored under `candidate.annotations[self.name]` for the
  module's evaluator to read. `SelfSimulationModule` implements this, and the
  framework invokes it during generation.

See `src/reflective/modules/self_simulation.py` for a complete example.

### Create a Module From the Abstract Base

To implement a module that is not a `PromptEvaluator` subclass — for example, a
deterministic check or a differently structured LLM-based evaluator — subclass
`ReflectiveModule` and implement `run()`. See [docs/modules.md](docs/modules.md)
for further explanation regarding `ReflectiveModule`.

## Full Example - Surprise Party Scenario

See `examples/selfsim_config.py` and `examples/run_selfsim_evaluation.py` for a runnable example that demonstrates the reflective pipeline with a full example. When in the root directory `reflective-llm`, use the following command to run the surprise party example:

> `PYTHONPATH=. python examples/run_selfsim_evaluation.py`

The model used for this example is `Qwen/Qwen2.5-0.5B-Instruct` with is around 3GB. The model can be changed to a different chat-tuned HuggingFace model in the configuration file `selfsim_config.py` under `MODEL_NAME`.

---

## Custom Vocabularies

This section illustrates how to customize the module slots to modify the classification vocabulary. 

The behaviour of `PromptEvaluator` is governed by slots — constructor parameters that default to module-level constants. To alter how the evaluator classifies, override `vocabulary` and `vocabulary_instructions`. The vocabulary is a `label → score` mapping over [0, 1]; the parser and scorer are derived from it automatically, so no changes to parsing are required.

```python
from reflective import PromptEvaluator

MY_VOCABULARY = {"PASS": 1.0, "WARN": 0.5, "FAIL": 0.0}
MY_INSTRUCTIONS = """
    - PASS: fully complies with the guideline
    - WARN: borderline/partial compliance
    - FAIL: clearly violates the guideline"""

class ComplianceModule(PromptEvaluator):
    name = "compliance"
    def __init__(self, guidelines, base_context):
        super().__init__(
            name="compliance",
            guidelines=guidelines,
            base_context=base_context,
            vocabulary=MY_VOCABULARY,
            vocabulary_instructions=MY_INSTRUCTIONS,
        )
```


> **Notes about resulting behaviour:**
>- The **highest-scoring label** (`PASS`) denotes compliance; any lower label is recorded as a failure and reduces the score.
>- The classification options presented to the model (`[Choose exactly one: PASS | WARN | FAIL]`) are generated automatically from the vocabulary keys.
>- The overall score is the **minimum** label score across all evaluated guidelines, compared against `min_score_threshold`.
>- `vocabulary_instructions` should describe exactly the labels in `vocabulary`. `vocabulary_instructions` defines each label for the model; `vocabulary` maps that same label to a score. This ensures that the provider does not return an incorrect label (as `vocabulary_instructions` is used to inform the provider about the classification labels). If an incorrect label is returned and the parser is unable to match it to a label in `vocabulary`, the evaluator treats the classification as if it is the best-rated label instead of raising an error. Mismatch flagging between the two slots is left up to future work.
>- The remaining constructor slots may be overridden in the same manner: `evaluation_prompt` (the complete template, which must retain the holes `{scenario}`, `{history}`, `{candidate_section}`, `{guidelines_section}`, and `{vocabulary_instructions}`), `generation_prompt`, and `history_window`. To alter the candidate or guidelines sections, override the methods `build_candidate_section` / `build_guidelines_section` in a subclass.

---

## Writing a Custom Provider

To use a model other than the built-in HuggingFace wrapper, subclass `Provider` and
implement `generate`, which returns a **list** of candidate strings. Returning more
than one candidate provides the reflective loop with alternatives to evaluate.

```python
from reflective import Provider

class OpenAIProvider(Provider):
    def __init__(self, model="gpt-4o-mini", n_candidates=2, **kwargs):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.n = n_candidates

    def generate(self, messages):
        resp = self.client.chat.completions.create(model=self.model, messages=messages, n=self.n)
        return [c.message.content for c in resp.choices]

    # generate_one() defaults to generate(messages)[0]; override where a single call is cheaper.
    # cleanup() is a no-op by default; override to release resources.
```

```python
from reflective import Reflective, EECModule, GuidelineRule

pipeline = Reflective(OpenAIProvider(), [EECModule(g, base_context=CTX)], base_context=CTX)
```

---

## Documentation

Complete documentation is provided in [`docs/`](docs/):

| Page | Contents |
|---|---|
| [docs/providers.md](docs/providers.md) | The `Provider` interface and `HFModel`. |
| [docs/modules.md](docs/modules.md) | The module base class, `PromptEvaluator`, `EECModule`, `SelfSimulationModule`. |
| [docs/framework.md](docs/framework.md) | `AgenticFramework` — the reflective loop. |
| [docs/pipeline.md](docs/pipeline.md) | `Reflective`, batch execution, and log capture. |
| [docs/dataclasses.md](docs/data-types.md) | The shared dataclasses and the configuration objects. |

---

## API reference

### Top-level (`from reflective import ...`)

| Name | Kind | Purpose |
|---|---|---|
| `Reflective` | class | The callable pipeline: `pipeline(messages, full_result=False, save=None)` and `run_batch(prompts)` |
| `AgenticFramework` | class | The orchestrator (wrapped by `Reflective`) |
| `Provider` | ABC | The base contract for model backends |
| `HFModel` | class | The built-in HuggingFace provider (requires the `[hf]` extra) |
| `ReflectiveModule` | ABC | The base contract for modules |
| `PromptEvaluator` | class | The slot-based evaluator base |
| `EECModule` | class | Built-in: assesses the immediate response |
| `SelfSimulationModule` | class | Built-in: assesses the response and its predicted trajectory |

### Dataclasses (`from reflective import ...`)

Defined in `reflective/dataclasses.py`. Full fields: [docs/dataclasses.md](docs/dataclasses.md).

| Name | Purpose |
|---|---|
| `FrameworkConfig` | `max_attempts`, `min_score_threshold`, `debug_mode` |
| `GenerationConfig` | `tokens`, `temperature`, `sampling`, `top_p`, `n_candidates`, `padding` |
| `GuidelineRule` | An expectation: `rule`, `context=""`, `examples=[]`, `fix_hint=""` |
| `TurnContext` | Read-only per-turn state: `user_input`, `conversation_history`, `base_context` |
| `ModuleVerdict` | A module's assessment: `score`, `passed`, `classifications`, `details`, `failures`, `artifact` |
| `ResponseCandidate` | A generated response and the information recorded about it |
| `TurnResult` | The complete result of one turn (with `winning_candidate()` and `to_dict()`) |


---

## Roadmap

The following directions are planned but not yet implemented:

- **Additional tutorials** — custom modules beyond the basics, including modules that compose different example behaviors.
- **Formal EEC** — derivation of expectations from world state (via `update_guidelines`) rather than static rules.
- **A richer reconciler** — detection of disagreement among modules and targeted re-evaluation, in place of minimum-score reconciliation.
- **Diagrams** — system and component diagrams to aid comprehension of the structure.
- **Common Mistakes** — The most common mistakes and pitfalls that may be encountered when using this package, and how to mitigate them.

---

## Citation

This package implements the architecture described in:

> P. Salmani and P. R. Lewis, "A Reflective Architecture for LLM-based Systems,"
> *2025 IEEE International Conference on Autonomic Computing and Self-Organizing
> Systems Companion (ACSOS-C)*, 2025. DOI: 10.1109/ACSOS-C66519.2025.00029.

The surprise-party scenario, the EEC and self-simulation modules, and the reflect-then-revise loop follow the architectural design proposed in this paper.
