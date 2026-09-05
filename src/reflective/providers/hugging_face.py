'''
Users can select a model from a list of providers. 
If their desired provider is not listed, they can add 
them as a class in the models.py. Provider implementation 
classes are concrete extensions of an abstract Provider class.
'''

###################### Imports ######################
from typing import List, Dict, Tuple, Optional
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field, replace

from .provider_base import Provider
from ..architecture.dataclasses import GenerationConfig

import logging

logger = logging.getLogger("reflective.providers.hf")

###################### HuggingFace ######################
class HFModel(Provider):
    """
    Generic HuggingFace causal-LM wrapper.

    Works with any chat-tuned model that ships an `apply_chat_template`
    (Gemma-IT, Llama-Instruct, Qwen-Chat, Mistral-Instruct, etc.).
    """

    def __init__(
        self,
        model_name: str,
        torch_dtype=torch.bfloat16, # Data type of torch.Tensor
        device_map: str = "auto", # Loads and distributes HF model onto available hardware (multiple GPUs, CPU, etc)
        trust_remote_code: bool = True, # When set to 'True', authorizes computer to run the HG model with a custom architecture. 
        generation_config: Optional[GenerationConfig] = None, # Sets the dictionary mapping configuration to specify the parameters that control the models behaviour
    ):
        super().__init__()
        if not model_name:
            raise ValueError(
                "model_name is empty. Set MODEL_NAME in the CONFIG section. "
                "Examples: 'google/gemma-2-9b-it', 'meta-llama/Llama-3.1-8B-Instruct', "
                "'Qwen/Qwen2.5-7B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.3'."
            )

        self.model_name = model_name

        logger.info(f"Loading {model_name}...")

        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            self.device = "cuda"
        else:
            logger.info("CUDA not available, using CPU")
            self.device = "auto"


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Model loaded on {self.device}")

        # Defaults — caller can override via `generation_config`.
        self.generation_config = generation_config or GenerationConfig()



    def _translate(self, config: GenerationConfig) -> Dict:
        """Map framework-neutral parameter names to HuggingFace names."""
        return {
            "max_new_tokens": config.tokens,
            "temperature": config.temperature,
            "do_sample": config.sampling,
            "top_p": config.top_p,
            "num_return_sequences": config.n_candidates,
            "pad_token_id": self.tokenizer.eos_token_id if config.padding is None else config.padding,
        }



    def generate(self, messages) -> List[str]:
        """Generate N candidate completions for a single prompt."""

        config = self._translate(self.generation_config)
        n = self.generation_config.n_candidates  

        logger.info(f"[LLM CALL] Generating {n} sequences")

        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **config,
            )

        input_length = inputs.input_ids.shape[1]
        responses = []
        for i in range(n):
            text = self.tokenizer.decode(
                outputs[i][input_length:], skip_special_tokens=True
            ).strip()
            responses.append(text)
            logger.info(f"[LLM RESPONSE {i+1}] Length: {len(text)} chars")
        return responses


    def cleanup(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()