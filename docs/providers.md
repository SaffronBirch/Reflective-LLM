# Providers — the model hosts

A **provider** runs inference to generate and evaluate candidates using a model host. Everything in the package — the reflective loop, the modules, the pipeline — works using a small, shared provider interface, so you can swap one model backend for another without changing the framework's architecture.

This page covers the two provider files: the abstract interface every provider follows
(`provider_base.py`) and the built-in HuggingFace provider (`hugging_face.py`) that enables the use of chat-tuned models.

## Contents

- [`provider_base.py`](#provider_basepy) — the `Provider` interface
- [`hugging_face.py`](#hugging_facepy) — `HFModel`, a ready-made HuggingFace provider


### Dataclass Used

**`GenerationConfig`** — the framework-neutral generation settings (how many
tokens, the temperature, how many candidates, etc). Each provider
translates these into the specific names its backend expects. Fields are listed on
the [Dataclasses](dataclasses.md#generationconfig) page.

---

## `provider_base.py`

### Overview

This file defines `Provider` (the base class every provider inherits from). `Provider` is *abstract* — so it is never used directly, but rather subclassed and formatted to fit the specific inference requirements for a given model host (HuggingFace, OpenAI, Ollama, etc).

### `class Provider`

The base class for all providers. It defines the three methods that candidate generation relies on, but the main method  `generate()` gets filled in by subclasses to account for differing inference call methods between providers.

#### **Constructor.** 

`Provider` does not define any constructors of its own. Each concrete provider defines constructor arguments independently.

#### **Methods**

**`generate(messages) -> list[str]`**

The one method every provider **must** implement. The framework calls it to get
the model's candidate replies for a prompt.

- `messages` — a list of chat messages, where each message is a dictionary, such as 
  `{"role": "user", "content": "..."}`.
- **Returns** a list of strings. Each string is one candidate reply. A provider
  may return any number of candidates, so the framework has more than one option to evaluate. The number of candidates is specified in the generation configuration [`GenerationConfig`](dataclasses.md#generationconfig).

It is marked `@abstractmethod`, which means Python will refuse to create a
provider that hasn't implemented it. 

**`generate_one(messages) -> str`**

A convenience method built on top of `generate`. It calls `generate` and returns
only the **first** reply.

- `messages` — the same chat-message list as above.
- **Returns** a single string (the first candidate).

Modules use this when they need exactly one answer from the model — for example,
when a module asks the model to classify a response as SAFE or RISKY, it wants
one verdict, not several. The base class provides it, and it works for any provider that implements `generate`.

**`cleanup() -> None`**

Called when you are finished with the provider, to release anything it is
holding. The base version does nothing. Providers that hold heavy resources — a
model loaded onto a GPU, an open network connection — override this to free them.
The pipeline does this by calling `cleanup()` upon exiting the `with Reflective(...) as pipeline` block.

---

## `hugging_face.py`

### Overview

`HFModel` is a ready-to-use provider that runs any chat-tuned model from
HuggingFace — Gemma, Llama, Qwen, Mistral, and similar. It handles loading the
model, formatting the chat prompt the way the model expects, generating replies,
and freeing memory afterward.

> Creating an `HFModel` imports `torch` and `transformers` and downloads the model the first time. That is why the package loads this provider only when `HFModel` itself is imported.

### Usage

```python
from reflective import HFModel, GenerationConfig

provider = HFModel(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    generation_config=GenerationConfig(n_candidates=2, temperature=0.9),
)

replies = provider.generate([{"role": "user", "content": "Hello!"}])
provider.cleanup()   # free the GPU memory when you're done
```

### `class HFModel(Provider)`

A concrete `Provider` backed by a HuggingFace causal language model.

#### **Constructor**

```python
HFModel(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    generation_config=None,
)
```

When you create an `HFModel`, it loads the tokenizer and the model, picks a
device (GPU if one is available, otherwise CPU), and — if the model has no
padding token — reuses its end-of-text token for padding. All of this happens
once, at construction time.

| Parameter           | Type                         | Default            | Description                                                                                  |
| ------------------- | ---------------------------- | ------------------ | --------------------------------------------------------------------------------------------- |
| `model_name`        | `str`                        | —                  | The HuggingFace model ID to load (e.g. `"google/gemma-2-9b-it"`). An empty value raises `ValueError` with examples. |
| `torch_dtype`       | `torch.dtype`                | `torch.bfloat16`   | The number format the model's weights are loaded in. `bfloat16` saves memory versus full precision. |
| `device_map`        | `str`                        | `"auto"`           | Tells `transformers` how to spread the model across the hardware. `"auto"` lets it decide.   |
| `trust_remote_code` | `bool`                       | `True`             | Allows loading models that ship their own custom code. Needed for some models.                |
| `generation_config` | `Optional[GenerationConfig]` | `None`             | The generation settings. If `None`, a default `GenerationConfig` is used.                     |

#### **Attributes Set During Construction**

- `model_name` — the ID you passed in.
- `device` — `"cuda"` if a GPU was found, otherwise `"auto"`.
- `tokenizer` — the loaded tokenizer.
- `model` — the loaded model.
- `generation_config` — the settings in use (custom if set, otherwise the default is used).

#### **Public Methods**

**`generate(messages) -> list[str]`**

Produces the candidate replies for one prompt. This is the method the framework
calls during generation. Step by step, it:

1. Reads how many replies to make from `generation_config.n_candidates`.
2. Formats `messages` into the exact prompt string the model expects, using the
   model's own chat template.
3. Turns that text into tokens (trimming anything past 4096 tokens).
4. Runs the model to generate the requested number of replies.
5. Decodes each reply back into text, drops the original prompt from the front,
   and strips whitespace.

- `messages` — the chat-message list.
- **Returns** a list of reply strings, one per candidate.

**`cleanup() -> None`**

Frees the memory the provider is holding. It deletes the model and tokenizer,
empties the GPU cache (if a GPU is in use), and runs Python's garbage collector.
Call this when you're done. The pipeline calls it by default.

#### **Internal Methods** (not part of the public interface, described for contributors)

**`_translate(config) -> dict`**

Converts the framework-neutral `GenerationConfig` into the keyword arguments
HuggingFace's own `generate` function expects. You never call this directly;
`generate` uses it. The mapping is:

| `GenerationConfig` field | HuggingFace argument                                             |
| ------------------------ | ---------------------------------------------------------------- |
| `tokens`                 | `max_new_tokens`                                                 |
| `temperature`            | `temperature`                                                    |
| `sampling`               | `do_sample`                                                      |
| `top_p`                  | `top_p`                                                          |
| `n_candidates`           | `num_return_sequences`                                           |
| `padding`                | `pad_token_id` (falls back to the tokenizer's end-of-text id when not set) |

### Notes

- **Writing your own provider.** Subclass `Provider`, implement `generate` (and
  `cleanup` if you hold resources), and translate your own `GenerationConfig`
  into your backend's parameters — that's what `_translate` does for HuggingFace.
- `HFModel` logs through the `reflective.providers.hf` logger, so its messages
  appear alongside the rest of the package's logging.
