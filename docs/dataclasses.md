# Data types & configuration

This page describes the seven dataclasses that are shared within the package, and one local dataclass that is exclusive to `SelfSimulationModule`. The shared dataclasses are gathered in one module, `reflective/dataclasses.py`.

## Contents

- [Data Types](#datatypes)
  - [`GuidelineRule`](#guidelinerule)
  - [`TurnContext`](#turncontext)
  - [`ModuleVerdict`](#moduleverdict)
  - [`ResponseCandidate`](#responsecandidate)
  - [`TurnResult`](#turnresult)
- [Configuration Types](#configurationtypes)
  - [`GenerationConfig`](#generationconfig)
  - [`FrameworkConfig`](#frameworkconfig)
- [Local Type](#localtype)
  - [`ConversationStep`](#conversationstep)

---

## Data Types

These five types are passed between the provider, the framework, and the modules. 

For each conversation turn, the general flow is as follows: a module checks a candidate against its `GuidelineRule`s using the read-only `TurnContext`, and returns a `ModuleVerdict`; the verdicts attach to a `ResponseCandidate`; and the winning candidate plus the full record are summed up in a `TurnResult`.

### `GuidelineRule`

One expectation a module checks against. Construct it as `GuidelineRule(rule, ...)`.

| Field       | Type        | Default | Description                                          |
| ----------- | ----------- | ------- | --------------------------------------------------- |
| `rule`      | `str`       | —       | The expectation itself, in plain words.             |
| `context`   | `str`       | `""`    | Extra background shown to the model when it judges. |
| `examples`  | `List[str]` | `[]`    | Optional examples of the rule.                      |
| `fix_hint`  | `str`       | `""`    | A hint on how to satisfy the rule, used in feedback.|

### `TurnContext`

The **read-only** facts shared with every module for one turn. It details the contextual information relevant for that turn to inform the provider. Construct it as `TurnContext(user_input, conversation_history, base_context)`.

| Field                  | Type          | Description                        |
| ---------------------- | ------------- | ------------------------------------ |
| `user_input`           | `str`         | The user's message this turn.        |
| `conversation_history` | `List[Dict]`  | The chat history so far.             |
| `base_context`         | `str`         | The scenario text.                   |

### `ModuleVerdict`

What a module returns after judging one candidate. It is returned by every module's
`run()` method. Construct it as `ModuleVerdict(score, passed, ...)` — the first two are required.

| Field             | Type          | Default | Description                                                                     |
| ----------------- | ------------- | ------- | -------------------------------------------------------------------------------- |
| `score`           | `float`       | —       | A score the framework compares against its threshold.                     |
| `passed`          | `bool`        | —       | Whether this module considers the candidate acceptable.                          |
| `classifications` | `List[str]`   | `[]`    | The module's own labels (e.g. SAFE / RISKY / VIOLATION). The framework never interprets these; they're for feedback and the trace. |
| `details`         | `Dict`        | `{}`    | Reasoning, the raw model answer, and anything else worth recording.              |
| `failures`        | `List[Dict]`  | `[]`    | Structured records of what went wrong, fed into the retry step.                  |
| `artifact`        | `Any`         | `None`  | An object the module produced for this candidate (stored under `annotations[name]`). |

### `ResponseCandidate`

A single generated response and all of its relevant details. Its verdicts and annotations
are keyed by module name. It is produced and scored by the [framework](framework.md).
Only `response` is required to construct a `ResponseCandidate`; the rest are default.

| Field                | Type                         | Default | Description                                             |
| -------------------- | ---------------------------- | ------- | ------------------------------------------------------- |
| `response`           | `str`                        | —       | The candidate's reply text.                             |
| `annotations`        | `Dict[str, Any]`             | `{}`    | Per-module artifacts (e.g. a predicted conversation).   |
| `verdicts`           | `Dict[str, ModuleVerdict]`   | `{}`    | Each module's verdict for this candidate.               |
| `score`              | `float`                      | `0.0`   | The reconciled score across all modules.                |
| `generation_attempt` | `int`                        | `1`     | Which attempt produced this candidate.                  |
| `sequence_id`        | `int`                        | `1`     | This candidate's position within its batch.             |
| `raw_llm_output`     | `str`                        | `""`    | The model's unparsed output.                            |
| `decision_reason`    | `str`                        | `""`    | Why the candidate was accepted or rejected.             |

### `TurnResult`

The summary of one whole turn — the object the pipeline returns when you ask for
`full_result=True`, and the shape that gets saved to disk. It is produced by the
[framework](framework.md). Only `prompt` is required to construct it; the rest are optional.

| Field                  | Type                       | Description                                               |
| ---------------------- | -------------------------- | --------------------------------------------------------- |
| `prompt`               | `str`                      | The turn's user input.                                     |
| `final_response`       | `str`                      | The reply returned to the caller.                          |
| `attempts_used`        | `int`                      | How many attempts ran.                                     |
| `success`              | `bool`                     | Whether a candidate cleared the threshold.                 |
| `fallback`             | `bool`                     | Whether the least-bad fallback was used.                   |
| `fallback_reason`      | `str`                      | An explanation when `fallback` is `True`.                  |
| `winning_score`        | `Optional[float]`          | The winning candidate's score.                             |
| `winning_attempt`      | `Optional[int]`            | The attempt that produced the winner.                      |
| `winning_sequence_id`  | `Optional[int]`            | The winner's position in its batch.                        |
| `all_candidates`       | `List[ResponseCandidate]`  | Every candidate generated across all attempts.             |
| `reasoning_trace`      | `List[Dict]`               | An ordered record of the framework's decisions.            |
| `feedback_context`     | `List[Dict]`               | The failures gathered and fed into later attempts.         |

#### **Methods**

**`winning_candidate() -> ResponseCandidate | None`**

Finds and returns the candidate that won this turn by matching
`winning_attempt` and `winning_sequence_id` against `all_candidates`. Returns
`None` if there's no match (for example, if nothing was generated).

**`to_dict() -> dict`**

Turns the whole result into a plain dictionary that can be written to JSON. On top
of the fields, it adds one derived entry, `winning_classifications` — a map of
each module's labels for the winning candidate — or `None` if there was no winner.

---

## Configuration Types

The two objects you create to control behaviour. Both are passed into
[`Reflective`](pipeline.md) (and `GenerationConfig` into a provider). Both are
dataclasses with all-default fields, so you can construct them with no arguments
and change only what you need.

### `GenerationConfig`

The generation settings, in framework-neutral names. Each provider translates
these into its own backend's terms (see the table under
[`HFModel`](providers.md#hugging_facepy)).

| Field          | Type            | Default | Description                                                 |
| -------------- | --------------- | ------- | -------------------------------------------------------------- |
| `tokens`       | `int`           | `512`   | The max numbe of  new tokens to generate.                               |
| `temperature`  | `float`         | `0.9`   | How randomized the output is (higher = more varied).              |
| `sampling`     | `bool`          | `True`  | Whether to sample (vs. always pick the top token).            |
| `top_p`        | `float`         | `0.9`   | Nucleus sampling: consider only the top slice of likely tokens.|
| `n_candidates` | `int`           | `2`     | How many replies to generate per prompt.                       |
| `padding`      | `Optional[int]` | `None`  | A pad-token id override; the provider picks a default when `None`. |

### `FrameworkConfig`

The reflective-loop settings, passed to both `AgenticFramework` and `Reflective`.

| Field                 | Type    | Default | What it does                                                        |
| --------------------- | ------- | ------- | ------------------------------------------------------------------- |
| `max_attempts`        | `int`   | `3`     | The max number of regeneration attempts before falling back.           |
| `min_score_threshold` | `float` | `0.7`   | The minimum score a candidate must reach to be accepted.         |
| `debug_mode`          | `bool`  | `True`  | Whether to print a detailed per-candidate breakdown to the console at runtime.  |

---

## Local Types

Some data types belong to a single component and are documented with it, because
they aren't part of the shared vocabulary:

- [`ConversationStep`](modules.md#self_simulationpy) — one predicted exchange in a
  self-simulation flow. Used only by `SelfSimulationModule`.
