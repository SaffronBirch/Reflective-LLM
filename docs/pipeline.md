# The Reflective Pipeline

`Reflective` is the front door to the whole package. It builds a pipeline with a
provider and any number of modules, then call it like a function to get a reflected reply.
It wraps the [framework](framework.md) and adds a simple call interface, batch runs, and optional saving.

## Contents

This page covers the reflective pipeline and a small helper that enables internal logging:
- [`pipeline.py`](#pipelinepy) — reflective pipeline
- [`capture.py`](#log-capture-_capturepy) — internal logging helpers


## `pipeline.py`

### Overview

`Reflective` is **quiet by default** — wrapping a call site adds no console
output unless you specify otherwise. You use it in one of three ways:

- **One reply:** `pipeline("...")` returns the final reply string.
- **The full result:** `pipeline("...", full_result=True)` returns
  [`TurnResult`](dataclasses.md#turnresult) as a per-turn results JSON file and the console logging to a `.log` file. 
- **A batch:** `pipeline.run_batch([...])` runs many prompts, each as its own
  conversation.

### Usage

```python
from reflective import Reflective, EECModule, HFModel, GuidelineRule

CTX = "You are a friend who knows about a surprise party on Saturday."

pipeline = Reflective(
    provider=HFModel(model_name="mistralai/Mistral-7B-Instruct-v0.3"),
    modules=[EECModule([GuidelineRule(rule="Do not reveal the party.")], base_context=CTX)],
    base_context=CTX,
)

reply = pipeline("Are you free Saturday?")                 # just the reply
result = pipeline("Are you free Saturday?", full_result=True)  # the full TurnResult
results = pipeline.run_batch(["Are you free?", "And Sunday?"])  # a batch
```

### `class Reflective`

#### **Constructor**

The `Reflective` class takes the same constructor arguments as `AgenticFramework`, but with two additional arguments: `output_dir` and `scenario_name`.  

```python
Reflective(
    provider,
    modules,
    base_context="",
    framework_config=None,
    slots=None,
    output_dir="./reflective_results",
    scenario_name="turn",
)
```

| Parameter          | Type                                            | Default                  | Description                                                        |
| ------------------ | ----------------------------------------------- | ------------------------ | ------------------------------------------------------------------- |
| `provider`         | `Provider`                                       | —                        | The model host used to generate and evaluate.                       |
| `modules`          | `List[ReflectiveModule]`                         | —                        | The modules run over every candidate.                               |
| `base_context`     | `str`                                            | `""`                     | The scenario text shared with modules and the generation prompt.    |
| `framework_config` | `Optional[FrameworkConfig]`                      | `None`                   | The loop settings. Framework defaults are used if `None`.           |
| `slots`            | `Optional[Dict[str, None \| str \| Callable]]`   | `None`                   | Overrides for the framework's generation-prompt slots.              |
| `output_dir`       | `str`                                            | `"./reflective_results"` | Where saved JSON and `.log` files go.                               |
| `scenario_name`    | `str`                                            | `"turn"`                 | A short tag used in saved filenames.                                |

#### **Attributes**

- `last_saved_path` — the path of the most recently saved results file, or `None` if nothing has been saved yet.

#### **Methods**

**`__call__(messages, full_result=False, save=None)`**

This method is called whenever the pipeline is used. It runs one full reflection-based turn for each prompt.

Runs one reflected turn. This is what happens when you call the pipeline like a
function.

- `messages` — either a plain string, or a chat-message list. If it's a list, the
  last user message is the turn's input and the whole list is the history.
- `full_result` — if `True`, return the whole `TurnResult`; if `False` (default),
  return just the reply string.
- `save` — whether to record the run and write a per-turn JSON + `.log`. If left
  as `None`, it follows `full_result` — Set it to `False` to only output the final response, or `True` to save all turn details and logging in addition to the response.
- **Returns** the reply string, or a `TurnResult`.

**`run_batch(prompts, save=True)`**

Runs several prompts, each as its **own independent conversation**, and (when
`save`) writes a single combined results file for the whole batch. This is the
pipeline's batch entry point.

- `prompts` — a list of prompt strings. Empty raises `ValueError`.
- `save` — if `True` (default), write one combined JSON + `.log` for the batch;
  if `False`, write nothing.
- **Returns** a list of `TurnResult` objects, one per prompt. When saved, the
  combined file's path is also stored on `last_saved_path`.

Note: this does not write one file per prompt. For separate per-prompt files,
call the pipeline once per prompt with `save=True` instead.

**`cleanup()` and the `with` block**

`cleanup()` releases the provider's resources by calling `provider.cleanup()`.
The pipeline also works as a context manager, so this happens automatically:

```python
with Reflective(provider, modules, base_context=CTX) as pipeline:
    reply = pipeline("hello")
# provider.cleanup() runs here automatically
```

`__enter__` returns the pipeline itself; `__exit__` calls `cleanup()`.

**Internal Methods** (not part of the public interface, described for contributors)

**`_last_user_message(messages) -> str` *(static)***

Returns the content of the most recent user message from the message list. Raises `ValueError` if there isn't one.

**`_print_batch_summary(records) -> None` *(static)***

Prints the end-of-batch summary — one line per prompt with its score, attempts,
and whether it fell back.

**`_save_turn(result, user_input, runtime_log) -> str`**

Saves one turn: it builds the results envelope (with a single `prompt`/`result`)
and writes it. Returns the JSON path.

**`_run_metadata(*, runtime_log, prompt=None, result=None, prompts=None, results=None) -> dict`**

Builds the dictionary that gets written to disk — the model name, scenario,
context, each module's guidelines, a timestamp, the configs, the results, and the
captured log. It's shared by both the per-turn and batch saves: pass the single
`prompt`/`result` for a turn, or the plural `prompts`/`results` for a batch.

**`_write_run(metadata) -> str`**

Writes a metadata dictionary to disk: a `<model>_<scenario>_<timestamp>.json`
file plus a matching `.log` file, both under `output_dir`. Records the JSON
path on `last_saved_path` and returns it.

---

## `capture.py`

### Overview

`capture.py` is an **internal** helper.
It captures everything printed and logged during a run so it can be saved with the
results. Both the per-turn save and the batch save use it, which is why their
saved transcripts look the same.

### Usage

```python
from reflective import capture_logging

with capture_logging(debug=False) as tee:
    run_something_that_prints_and_logs()

transcript = tee.getvalue()   # everything printed + logged in the block
```

### `capture_logging(debug)` *(context manager)*

Captures console and log output for the duration of the run. It redirects
`sys.stdout` through a `TeeLogger` and attaches a handler to the `reflective`
logger, then restores both when the run ends.

- `debug` — if `True`, also capture `DEBUG`-level log messages; otherwise capture
  from `INFO` up.
- **Yields** a `TeeLogger`; call its `.getvalue()` afterward for the full
  transcript.

### `class TeeLogger`

Copies everything written to a stream (normally `sys.stdout`) into an in-memory
buffer, so the output can be shown live *and* saved.

#### **Constructor.** 

`TeeLogger(stream)` — wraps the given stream and starts an empty buffer.

#### **Methods**

- `write(text) -> None` — writes the text to the wrapped stream and appends
  it to the buffer.
- `flush() -> None` — flushes the wrapped stream.
- `getvalue() -> str` — returns everything captured so far, as one string.

### `class _StdoutLogHandler`

An internal logging handler. Its one method, `emit(record)`, writes each log
message to whatever `sys.stdout` is at that moment — which is the `TeeLogger`
while a capture block is running.

## Notes

- **Filenames won't collide.** Saved files use microsecond-precision timestamps,
  so saving many turns in a tight loop won't overwrite earlier ones.
- **Same shape, turn or batch.** Per-turn and batch saves share one builder, so
  the files look the same apart from `prompt`/`result` (a turn) versus
  `prompts`/`results` (a batch).
