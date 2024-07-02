import asyncio
import os
from copy import deepcopy
from typing import List, Optional

import numpy as np
import openai
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedTokenizer,
)

from .attribution_metrics import (
    cosine_similarity_attribution,
    token_prob_attribution,
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
        request_chunksize: Optional[int] = 1000,
    ):
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.openai_model = openai_model or "gpt-3.5-turbo"

        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")
        self.token_embeddings = (
            token_embeddings
            or GPT2LMHeadModel.from_pretrained("gpt2").transformer.wte.weight.detach().numpy()
        )
        self.request_chunksize = request_chunksize

    async def get_chat_completion(self, input: str) -> openai.types.chat.chat_completion.Choice:
        response = await self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": input}],
            temperature=0.0,
            seed=0,
            logprobs=True,
            top_logprobs=20,
        )
        return response.choices[0]

    async def compute_attributions(self, input_text: str, **kwargs):
        perturbation_strategy: PerturbationStrategy = kwargs.get(
            "perturbation_strategy", FixedPerturbationStrategy()
        )
        attribution_strategies: List[str] = kwargs.get(
            "attribution_strategies", ["cosine", "prob_diff"]
        )
        logger: ExperimentLogger = kwargs.get("logger", None)
        perturb_word_wise: bool = kwargs.get("perturb_word_wise", False)
        ignore_output_token_location: bool = kwargs.get("ignore_output_token_location", True)

        original_output = await self.get_chat_completion(input_text)
        remaining_output = deepcopy(original_output)

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
            words = [" " + w for w in input_text.split()]
            words[0] = words[0][1:]
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

        tasks = []
        perturbations = []
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

            # Create task for the perturbed input
            tasks.append(asyncio.create_task(self.get_chat_completion(perturbed_input)))
            perturbations.append(
                {
                    "input": perturbed_input,
                    "unit_tokens": unit_tokens,
                    "replaced_token_ids": replacement_token_ids,
                }
            )

        # Get the output logprobs for the perturbed inputs
        if self.request_chunksize is not None and len(tasks) > self.request_chunksize:
            outputs = []
            for idx in tqdm(range(0, len(tasks), self.request_chunksize), desc=f"Sending {self.request_chunksize:.0f} concurrent requests at a time"):
                batch = [tasks[i] for i in range(idx, min(idx + self.request_chunksize, len(tasks)))]
                outputs.extend(await asyncio.gather(*batch))
                await asyncio.sleep(0.1)
        else:
            outputs = await asyncio.gather(*tasks)

        for perturbation, perturbed_output in zip(perturbations, outputs):
            if ignore_output_token_location:
                all_top_logprobs = []
                all_toks = []
                for ptl in perturbed_output.logprobs.content:
                    all_top_logprobs.extend([tl.logprob for tl in ptl.top_logprobs])
                    all_toks.extend([tl.token for tl in ptl.top_logprobs])

                sorted_indexes = sorted(
                    range(len(all_top_logprobs)), key=all_top_logprobs.__getitem__, reverse=True
                )
                all_toks = [all_toks[s] for s in sorted_indexes]
                all_top_logprobs = [all_top_logprobs[s] for s in sorted_indexes]

                for otl in remaining_output.logprobs.content:
                    if otl.token in all_toks:
                        new_lp = all_top_logprobs[all_toks.index(otl.token)]

                    else:
                        new_lp = -100

                    otl.logprob = new_lp
                    for tl in otl.top_logprobs:
                        if tl.token == otl.token:
                            tl.logprob = new_lp

                remaining_output.message.content = perturbed_output.message.content
                perturbed_output = remaining_output

            for attribution_strategy in attribution_strategies:
                if attribution_strategy == "cosine":
                    sentence_attr, attributed_tokens, token_attributions = (
                        cosine_similarity_attribution(
                            original_output.message.content,
                            perturbed_output.message.content,
                            self.token_embeddings,
                            self.tokenizer,
                        )
                    )
                elif attribution_strategy == "prob_diff":
                    sentence_attr, attributed_tokens, token_attributions = token_prob_attribution(
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
                                perturbation["input"],
                                perturbed_output.message.content,
                            )
            unit_offset += len(unit_tokens)

        if logger:
            logger.log_perturbation(
                i,
                self.tokenizer.decode(perturbation["replaced_token_ids"], skip_special_tokens=True)[
                    0
                ],
                perturbation_strategy,
                input_text,
                original_output.message.content,
                perturbation["input"],
                perturbed_output.message.content,
            )
            logger.stop_experiment()
