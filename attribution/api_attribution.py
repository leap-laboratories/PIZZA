import asyncio
import itertools
import os
import statistics
from copy import deepcopy
from typing import Any, List, Optional

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
    NEAR_ZERO_PROB,
    cosine_similarity_attribution,
    token_prob_attribution,
)
from .base import BaseAsyncLLMAttributor
from .experiment_logger import ExperimentLogger
from .token_perturbation import (
    FixedPerturbationStrategy,
    PerturbationStrategy,
)

load_dotenv()

DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"


class OpenAIAttributor(BaseAsyncLLMAttributor):
    def __init__(
        self,
        openai_model: str = DEFAULT_OPENAI_MODEL,
        openai_api_key: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        token_embeddings: Optional[np.ndarray] = None,
        request_chunksize: Optional[int] = 50,
    ):
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.openai_model = openai_model

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

    def make_output_location_invariant(self, original_output, perturbed_output):
        # Making a copy of the original output, so we can update it with the perturbed output log probs, wherever a token from the unperturned output is found in the perturbed output.
        location_invariant_output = deepcopy(original_output)

        # Get lists of all tokens and their logprobs (including top 20 in each output position) in the perturbed output
        all_top_logprobs = []
        all_tokens = []
        for perturbed_token in perturbed_output.logprobs.content:
            all_top_logprobs.extend(
                [token_logprob.logprob for token_logprob in perturbed_token.top_logprobs]
            )
            all_tokens.extend(
                [token_logprob.token for token_logprob in perturbed_token.top_logprobs]
            )

        # Sorting the tokens and logprobs by logprob in descending order. This is because .index gets the first occurence of a token in the list, and we want to get the highest logprob for each token.
        sorted_indexes = sorted(
            range(len(all_top_logprobs)), key=all_top_logprobs.__getitem__, reverse=True
        )
        all_tokens_sorted = [all_tokens[s] for s in sorted_indexes]
        all_top_logprobs_sorted = [all_top_logprobs[s] for s in sorted_indexes]

        # Now, for each token in the original output, if it is found in the perturbed output , update the logprob in the original output with the logprob from the perturbed output.
        # Otherwise, set the logprob to a near zero value.

        for unperturbed_token in location_invariant_output.logprobs.content:
            if unperturbed_token.token in all_tokens_sorted:
                perturbed_logprob = all_top_logprobs_sorted[
                    all_tokens_sorted.index(unperturbed_token.token)
                ]
            else:
                perturbed_logprob = NEAR_ZERO_PROB

            # Update the main token logprob
            unperturbed_token.logprob = perturbed_logprob

            # Update the same token logprob in the top 20 logprobs (duplicate information, but for consistency with the original output structure / OpenAI format)
            for top_logprob in unperturbed_token.top_logprobs:
                if top_logprob.token == unperturbed_token.token:
                    top_logprob.logprob = perturbed_logprob

        # And update the message content
        location_invariant_output.message.content = perturbed_output.message.content

        # Now the perturbed output contains the same tokens as the original output, but with the logprobs from the perturbed output.
        return location_invariant_output

    def get_units(self, input_text, chunksize, **kwargs):

        perturb_word_wise: bool = kwargs.get("perturb_word_wise", False)

        # A unit is either a word or a single token, depending on the value of `perturb_word_wise`
        if perturb_word_wise:
            words = [" " + w for w in input_text.split()]
            words[0] = words[0][1:]
            tokens_per_unit = [self.tokenizer.tokenize(word) for word in words]
        else:
            tokens_per_unit = [[token] for token in self.tokenizer.tokenize(input_text)]

        return tokens_per_unit

    async def hierarchical_perturbation(self, input_text: str, init_chunksize: int, threshold: float = 0.5, perturb_word_wise: bool = False, logger: Optional[ExperimentLogger] = None, **kwargs):
        
        tokens = self.get_units(input_text, chunksize=1, perturb_word_wise=perturb_word_wise)
        token_count = len(tokens)

        perturbation_strategy: PerturbationStrategy = kwargs.get(
            "perturbation_strategy", FixedPerturbationStrategy()
        )

        attribution_strategies: List[str] = kwargs.get(
            "attribution_strategies", ["cosine", "prob_diff"]
        )

        # Initialize mask with initial chunks
        masks = []
        for start in range(0, token_count, init_chunksize):
            end = min(start + init_chunksize, token_count)
            mask = np.zeros(token_count, dtype=bool)
            mask[start:end] = True
            masks.append(mask)

        original_output = await self.get_chat_completion(input_text)
        if logger:
            logger.start_experiment(
                input_text,
                original_output.message.content,
                perturbation_strategy,
                perturb_word_wise,
            )

        final_scores = np.zeros(token_count)
        total_llm_calls = 1

        while masks:
            new_masks = []
            perturbation_scores = []
            perturbations = []
            for mask in masks:
                perturbed_units = [token if not mask[i] else perturbation_strategy.replacement_token for i, token in enumerate(tokens)]
                # TODO: Check this is correct unit > token conversion  
                perturbed_tokens = ["".join(unit) if isinstance(unit, list) else unit for unit in perturbed_units]
                # TODO: Make list instead of dict (compute_attribution_chunks change required)
                perturbations.append(
                    {
                        "input": self.tokenizer.convert_tokens_to_string(perturbed_tokens),
                        "unit_tokens": [tokens[i] for i in np.where(mask)[0]],
                        "token_idx": np.where(mask)[0].tolist(),
                    }
                )
                
            outputs = await self.compute_attribution_chunks(perturbations)
            chunk_scores = self.get_scores(outputs, original_output, **kwargs)
            total_llm_calls += len(outputs)

            if logger:
                for perturbation, output, score in zip(perturbations, outputs, chunk_scores):
                    for unit_token, token_id in zip(perturbation["unit_tokens"], perturbation["token_idx"]):

                        for attribution_strategy, attr_result in score.items():
                            
                            logger.log_input_token_attribution(
                                attribution_strategy,
                                token_id,
                                "".join(unit_token),
                                float(attr_result["sentence_attribution"]),
                            )
                            for j, attr_score in enumerate(attr_result["token_attributions"]):
                                logger.log_token_attribution_matrix(
                                    attribution_strategy,
                                    token_id,
                                    j,
                                    attr_result["attributed_tokens"][j],
                                    attr_score.squeeze(),
                                    perturbation["input"],
                                    output.message.content,
                                )

                    logger.log_perturbation(
                        len(perturbation["unit_tokens"]) - 1,
                        perturbation_strategy.replacement_token,
                        str(perturbation_strategy),
                        input_text,
                        original_output.message.content,
                        perturbation["input"],
                        output.message.content,
                    )

            for mask, score in zip(masks, chunk_scores):
                attr_score = score[attribution_strategies[0]]["sentence_attribution"]
                perturbation_scores.append(attr_score)
                final_scores[mask] = attr_score

            midrange_score = (np.max(perturbation_scores) + np.min(perturbation_scores)) / 2
            if midrange_score < 0.01:
                break
            
            for mask, score in zip(masks, perturbation_scores):
                if (score >= midrange_score or score > threshold) and mask.sum() > 1:
                    # Split the chunk in half
                    indices = np.where(mask)[0]
                    mid = len(indices) // 2
                    
                    mask1 = np.zeros(token_count, dtype=bool)
                    mask2 = np.zeros(token_count, dtype=bool)
                    
                    mask1[indices[:mid]] = True
                    mask2[indices[mid:]] = True
                    
                    new_masks.append(mask1)
                    new_masks.append(mask2)

            masks = new_masks

        if logger:
            logger.df_token_attribution_matrix = logger.df_token_attribution_matrix.drop_duplicates(subset=["exp_id", "input_token_pos", "output_token"], keep="last").sort_values(by=["input_token_pos", "output_token_pos"]).reset_index(drop=True)
            logger.df_input_token_attribution = logger.df_input_token_attribution.drop_duplicates(subset=["exp_id","input_token_pos"], keep="last").sort_values(by="input_token_pos").reset_index(drop=True)
            logger.stop_experiment(num_llm_calls=total_llm_calls)

        return final_scores.tolist()

    def get_scores(self, perturbed_output, original_output, **kwargs) -> list[dict[str, Any]]:
        attribution_strategies: List[str] = kwargs.get(
            "attribution_strategies", ["cosine", "prob_diff"]
        )

        ignore_output_token_location: bool = kwargs.get("ignore_output_token_location", True)

        results = []
        for output in perturbed_output:
            if ignore_output_token_location:
                output = self.make_output_location_invariant(
                    original_output, output
                )
            
            scores = {}
            for attribution_strategy in attribution_strategies:
                if attribution_strategy == "cosine":
                    sentence_attr, attributed_tokens, token_attributions = (
                        cosine_similarity_attribution(
                            original_output.message.content,
                            output.message.content,
                            self.token_embeddings,
                            self.tokenizer,
                        )
                    )
                elif attribution_strategy == "prob_diff":
                    sentence_attr, attributed_tokens, token_attributions = token_prob_attribution(
                        original_output.logprobs, output.logprobs
                    )
                else:
                    raise ValueError(f"Unknown attribution strategy: {attribution_strategy}")
                
                scores[attribution_strategy] = {
                    "sentence_attribution": sentence_attr,
                    "attributed_tokens": attributed_tokens,
                    "token_attributions": token_attributions,
                }

            results.append(scores)                

        return results

    async def compute_attribution_chunks(self, perturbations: list[dict[str, Any]]) -> list[openai.types.CompletionChoice]:

        tasks = [asyncio.create_task(self.get_chat_completion(perturbation["input"])) for perturbation in perturbations]

        # Get the output logprobs for the perturbed inputs
        if self.request_chunksize is not None and len(tasks) > self.request_chunksize:
            outputs = []
            for idx in tqdm(
                range(0, len(tasks), self.request_chunksize),
                desc=f"Sending {self.request_chunksize:.0f} concurrent requests at a time",
            ):
                batch = [
                    tasks[i] for i in range(idx, min(idx + self.request_chunksize, len(tasks)))
                ]
                outputs.extend(await asyncio.gather(*batch))
                await asyncio.sleep(0.1)
        else:
            outputs = await asyncio.gather(*tasks)

        return outputs

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

        if logger:
            logger.start_experiment(
                input_text,
                original_output.message.content,
                perturbation_strategy,
                perturb_word_wise,
            )

        # A unit is either a word or a single token, depending on the value of `perturb_word_wise`
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
                    "token_id": i_unit,
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
                perturbed_output = self.make_output_location_invariant(
                    original_output, perturbed_output
                )

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
                    logger.log_input_token_attribution(
                        attribution_strategy,
                        perturbation["token_id"],
                        "".join(perturbation["unit_tokens"]),
                        float(sentence_attr),
                    )
                    for j, attr_score in enumerate(token_attributions):
                        logger.log_token_attribution_matrix(
                            attribution_strategy,
                            perturbation["token_id"],
                            j,
                            attributed_tokens[j],
                            attr_score.squeeze(),
                            perturbation["input"],
                            perturbed_output.message.content,
                        )

        if logger:
            logger.log_perturbation(
                len(perturbation["unit_tokens"]) - 1,
                self.tokenizer.decode(perturbation["replaced_token_ids"], skip_special_tokens=True)[
                    0
                ],
                str(perturbation_strategy),
                input_text,
                original_output.message.content,
                perturbation["input"],
                perturbed_output.message.content,
            )
            logger.stop_experiment(num_llm_calls=len(outputs))
