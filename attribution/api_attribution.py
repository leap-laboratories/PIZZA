import os
from typing import List, Optional

import numpy as np
import openai
from dotenv import load_dotenv
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedTokenizer,
)

from .attribution_metrics import (
    cosine_similarity_attribution,
    token_displacement,
    token_prob_difference,
)
from .base import BaseLLMAttributor
from .experiment_logger import ExperimentLogger
from .token_perturbation import (
    FixedPerturbationStrategy,
    PerturbationStrategy,
)

load_dotenv()


class OpenAIAttributor(BaseLLMAttributor):
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        token_embeddings: Optional[np.ndarray] = None,
    ):
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.openai_model = openai_model or "gpt-3.5-turbo"

        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")
        self.token_embeddings = token_embeddings or GPT2LMHeadModel.from_pretrained("gpt2").transformer.wte.weight.detach().numpy()

    def get_chat_completion(self, input: str) -> openai.types.chat.chat_completion.Choice:
        
        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": input}],
            temperature=0.0,
            seed=0,
            logprobs=True,
            top_logprobs=20,
        )
        return response.choices[0]

    def compute_attributions(self, input_text: str, **kwargs):
        perturbation_strategy: PerturbationStrategy = kwargs.get(
            "perturbation_strategy", FixedPerturbationStrategy()
        )
        attribution_strategies: List[str] = kwargs.get(
            "attribution_strategies", ["cosine", "prob_diff", "token_displacement"]
        )
        logger: ExperimentLogger = kwargs.get("logger", None)
        perturb_word_wise: bool = kwargs.get("perturb_word_wise", False)

        original_output = self.get_chat_completion(input_text)

        if logger:
            logger.start_experiment(
                input_text,
                original_output.message.content,
                perturbation_strategy,
                perturb_word_wise,
            )

        # A unit is either a word or a single token, depending on the value of `perturb_word_wise`
        unit_offset = 0
        if perturb_word_wise:
            words = input_text.split()
            tokens_per_unit = [self.tokenizer.tokenize(word) for word in words]
            token_ids_per_unit = [
                self.tokenizer.encode(word, add_special_tokens=False) for word in words
            ]
        else:
            tokens_per_unit = [[token] for token in self.tokenizer.tokenize(input_text)]
            token_ids_per_unit = [
                [token_id]
                for token_id in self.tokenizer.encode(input_text, add_special_tokens=False)
            ]

        for i_unit, unit_tokens in enumerate(tokens_per_unit):
            replacement_token_ids = [
                perturbation_strategy.get_replacement_token(token_id)
                for token_id in token_ids_per_unit[i_unit]
            ]

            # Replace the current word with the new tokens
            left_token_ids = [
                token_id
                for unit_token_ids in token_ids_per_unit[:i_unit]
                for token_id in unit_token_ids
            ]
            right_token_ids = [
                token_id
                for unit_token_ids in token_ids_per_unit[i_unit + 1 :]
                for token_id in unit_token_ids
            ]
            perturbed_input = self.tokenizer.decode(
                left_token_ids + replacement_token_ids + right_token_ids, skip_special_tokens=True
            )

            # Get the output logprobs for the perturbed input
            perturbed_output = self.get_chat_completion(perturbed_input)

            for attribution_strategy in attribution_strategies:
                attributed_tokens = [
                    token_logprob.token for token_logprob in original_output.logprobs.content
                ]
                if attribution_strategy == "cosine":
                    sentence_attr, token_attributions = cosine_similarity_attribution(
                        original_output, perturbed_output, self.token_embeddings, self.tokenizer
                    )
                elif attribution_strategy == "prob_diff":
                    sentence_attr, attributed_tokens, token_attributions = token_prob_difference(
                        original_output.logprobs, perturbed_output.logprobs
                    )
                elif attribution_strategy == "token_displacement":
                    sentence_attr, attributed_tokens, token_attributions = token_displacement(
                        original_output.logprobs, perturbed_output.logprobs
                    )
                else:
                    raise ValueError(f"Unknown attribution strategy: {attribution_strategy}")

                if logger:
                    for i, unit_token in enumerate(unit_tokens):
                        logger.log_input_token_attribution(
                            attribution_strategy,
                            unit_offset + i,
                            unit_token,
                            float(sentence_attr),
                        )
                        for j, attr_score in enumerate(token_attributions):
                            logger.log_token_attribution_matrix(
                                attribution_strategy,
                                unit_offset + i,
                                j,
                                attributed_tokens[j],
                                attr_score.squeeze(),
                                perturbed_input,
                                perturbed_output.message.content,
                            )
            unit_offset += len(unit_tokens)

        if logger:
            logger.log_perturbation(
                i,
                self.tokenizer.decode(replacement_token_ids, skip_special_tokens=True)[0],
                perturbation_strategy,
                input_text,
                original_output.message.content,
                perturbed_input,
                perturbed_output.message.content,
            )
            logger.stop_experiment()
