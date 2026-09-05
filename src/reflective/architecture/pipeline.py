'''
Reflective — add a reflective validation loop to an app in one line.

    from reflective import Reflective

    pipeline = Reflective(provider, modules, base_context=BASE_CONTEXT)
    reply = pipeline(messages)      

Quiet by default: the package logs through the "reflective" logger and attaches
only a NullHandler, so wrapping a call site adds no console output unless the
host app opts in.

Saving: a call made with ``full_result=True`` also captures the run's console
+ logging output and writes a per-turn results JSON and a ``.log`` sidecar.
Pass ``save=False`` to get the TurnResult object without touching the
filesystem, or ``save=True`` to save even when returning just the reply string.

Batch: ``pipeline.run_batch(prompts)`` runs each prompt as an independent
conversation and writes ONE aggregated JSON + ``.log`` for the whole batch.
It is the single entry point for what a standalone batch runner used to do;
the capture machinery it shares with the per-turn path lives in
``reflective/capture.py``.
'''

###################### Imports ######################
import os
import json
import datetime
from dataclasses import asdict, is_dataclass
from typing import Dict, List, Optional, Union

from ..modules.module_base import ReflectiveModule
from ..providers.provider_base import Provider

from .capture import capture_logging
from .framework import AgenticFramework
from .dataclasses import FrameworkConfig, TurnResult, GuidelineRule, GenerationConfig


###################### Wrapper ######################
class Reflective:
    """A thin, callable wrapper around AgenticFramework."""

    def __init__(
        self,
        provider: Provider,
        modules: List[ReflectiveModule],
        base_context: str = "",
        framework_config: Optional[FrameworkConfig] = None,
        slots: Optional[Dict[str, Union[None, str, callable]]] = None,
        output_dir: str = "./reflective_results",
        scenario_name: str = "turn",
    ):
        self.provider = provider
        self.modules = modules
        self.base_context = base_context
        self.output_dir = output_dir
        self.scenario_name = scenario_name
        self.last_saved_path: Optional[str] = None  # path of the most recent save
        self._framework = AgenticFramework(
            provider=provider,
            modules=modules,
            base_context=base_context,
            framework_config=framework_config,
            slots=slots,
        )

    def __call__(
        self,
        messages: Union[str, List[Dict[str, str]]],
        full_result: bool = False,
        save: Optional[bool] = None,
    ) -> Union[str, TurnResult]:
        """
        Run one reflected turn. `messages` is a chat-message list (its last user
        message is the turn's input; the whole list is the history) or a bare
        string. Returns the final reply, or the full TurnResult when
        full_result=True.

        save    - whether to capture the run and write a per-turn JSON + .log to
                  ``output_dir``. Defaults to ``full_result`` (so asking for the
                  full object also saves the run). Set False to keep the pure,
                  quiet, no-filesystem behaviour; set True to save even when
                  returning just the reply string.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        user_input = self._last_user_message(messages)
        should_save = full_result if save is None else save

        if not should_save:
            # Pure, quiet path — unchanged from the original wrapper.
            result = self._framework.main_chat_handler(user_input, messages)
            return result if full_result else result.final_response

        # Capture the run's console + logging output, then persist it. Shares
        # the same capture machinery as run_batch (see reflective/_capture.py),
        # so per-turn and batch transcripts are identical in format.
        debug = self._framework.framework_config.debug_mode
        with capture_logging(debug) as tee:
            result = self._framework.main_chat_handler(user_input, messages)

        self._save_turn(result, user_input, tee.getvalue())
        return result if full_result else result.final_response

    @staticmethod
    def _last_user_message(messages: List[Dict[str, str]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return message["content"]
        raise ValueError("messages contains no user message to respond to")

    ###################### Batch running ######################
    def run_batch(self, prompts: List[str], save: bool = True) -> List[TurnResult]:
        """
        Run every prompt as an independent conversation and (when ``save``)
        write ONE aggregated JSON + ``.log`` for the whole batch — the shape and
        naming a standalone batch runner produced. This is the pipeline's batch
        entry point; there is no separate runner.

        Returns the list of TurnResults; the aggregated JSON path is also stored
        on ``last_saved_path``. Per-turn files are never written here — for those,
        call the pipeline once per prompt with ``save=True``.
        """
        if not prompts:
            raise ValueError("prompts is empty — add at least one test prompt.")

        debug = self._framework.framework_config.debug_mode
        with capture_logging(debug) as tee:
            results: List[TurnResult] = []
            records: List[Dict] = []
            for idx, prompt_text in enumerate(prompts, 1):
                print(f"\n{'#'*70}")
                print(f"# PROMPT {idx}/{len(prompts)}: {prompt_text}")
                print(f"{'#'*70}")

                # Each prompt is its own conversation — same history shape the
                # __call__ path builds for a bare string.
                result = self._framework.main_chat_handler(
                    prompt_text, [{"role": "user", "content": prompt_text}]
                )
                results.append(result)

                record = result.to_dict()
                record["prompt_index"] = idx
                records.append(record)

                print(f"\nRESULT {idx}: score={result.winning_score}, "
                      f"attempts={result.attempts_used}, success={result.success}")
                print(f"   → {result.final_response[:120]}")

            # Printed inside the capture block so the summary lands in both the
            # JSON runtime_log and the .log sidecar.
            self._print_batch_summary(records)

        if save:
            self._write_run(self._run_metadata(
                prompts=prompts, results=records, runtime_log=tee.getvalue()))
        return results

    @staticmethod
    def _print_batch_summary(records: List[Dict]) -> None:
        print(f"\n{'='*70}\n SUMMARY:")
        for r in records:
            status = "" if r["success"] else "⚠️ fallback"
            print(f"  {r['prompt_index']:2d}. [{status}] score={r['winning_score']} "
                  f"attempts={r['attempts_used']} | {r['prompt'][:60]}")

    ###################### Saving ######################
    def _save_turn(self, result: TurnResult, user_input: str, runtime_log: str) -> str:
        """
        Write one turn's result + captured log to disk. Same envelope as a batch
        save, with one ``prompt``/``result`` instead of ``prompts``/``results``.
        Returns the JSON path.
        """
        return self._write_run(self._run_metadata(
            prompt=user_input, result=result.to_dict(), runtime_log=runtime_log))

    def _run_metadata(
        self,
        *,
        runtime_log: str,
        prompt: Optional[str] = None,
        result: Optional[Dict] = None,
        prompts: Optional[List[str]] = None,
        results: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        The single source of the results-file envelope, shared by per-turn and
        batch saves. Pass either the singular pair (``prompt``/``result``) or the
        plural pair (``prompts``/``results``); the file's key names follow suit.
        """
        model_name = getattr(self.provider, "model_name", self.provider.__class__.__name__)
        meta: Dict = {
            "model": model_name,
            "scenario": self.scenario_name,
            "base_context": self._framework.base_context,
            "modules": {
                m.name: [asdict(e) for e in getattr(m, "guidelines", [])]
                for m in self.modules
            },
        }
        # Input side: one prompt (turn) or many (batch).
        meta["prompts" if prompts is not None else "prompt"] = (
            prompts if prompts is not None else prompt)
        # Microsecond precision so saves in a tight loop don't collide/overwrite.
        meta["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        meta["framework_config"] = asdict(self._framework.framework_config)
        meta["generation_config"] = (
            asdict(self.provider.generation_config)
            if is_dataclass(getattr(self.provider, "generation_config", None)) else {})
        # Output side: one result (turn) or many (batch).
        meta["results" if results is not None else "result"] = (
            results if results is not None else result)
        meta["runtime_log"] = runtime_log
        return meta

    def _write_run(self, metadata: Dict) -> str:
        """Write a metadata envelope as ``<model>_<scenario>_<ts>.json`` plus a
        matching ``.log`` sidecar; record the JSON path on ``last_saved_path``."""
        os.makedirs(self.output_dir, exist_ok=True)
        safe_model_name = metadata["model"].replace("/", "_")
        output_file = os.path.join(
            self.output_dir,
            f"{safe_model_name}_{self.scenario_name}_{metadata['timestamp']}.json",
        )
        with open(output_file, "w") as f:
            json.dump(metadata, f, indent=2)

        log_file = output_file.replace(".json", ".log")
        with open(log_file, "w") as f:
            f.write(metadata["runtime_log"])

        print(f"Saved → {output_file}")
        print(f"Full runtime log → {log_file}")
        self.last_saved_path = output_file
        return output_file

    def cleanup(self) -> None:
        self.provider.cleanup()

    # Context-manager support: `with Reflective(...) as pipeline:` cleans up.
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()