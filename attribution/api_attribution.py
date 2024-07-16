import asyncio
import os
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import openai
import pandas as pd
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
    PerturbedLLMInput,
)
from .types import StrictChoice

load_dotenv()

DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
REQUEST_DELAY = 0.1
MIN_MIDRANGE_THRESHOLD = 0.01


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
            or GPT2LMHeadModel.from_pretrained("gpt2").transformer.wte.weight.detach().numpy() # type: ignore
        )
        self.request_chunksize = request_chunksize

    async def compute_attributions(
        self,
        original_input: str,
        perturbation_strategy: PerturbationStrategy = FixedPerturbationStrategy(),
        attribution_strategies: list[str] = ["cosine", "prob_diff"],
        logger: Optional[ExperimentLogger] = None,
        perturb_word_wise: bool = False,
        ignore_output_token_location: bool = True,
    ):
        original_output = await self._get_chat_completion(original_input)

        if logger:
            logger.start_experiment(
                original_input,
                original_output.message.content,
                perturbation_strategy,
                perturb_word_wise,
            )

        units, tokens_per_unit, ids_per_unit = self._get_units(original_input, perturb_word_wise=perturb_word_wise)

        perturbations: list[PerturbedLLMInput] = []
        for i_unit, unit in enumerate(units):
            perturbed_unit = [
                self.tokenizer.decode(perturbation_strategy.get_replacement_token(token_id)).strip()
                for token_id in ids_per_unit[i_unit]
            ]

            perturbed_input = deepcopy(tokens_per_unit)
            perturbed_input[i_unit] = perturbed_unit

            perturbation = PerturbedLLMInput(
                input_units=perturbed_input, 
                masked_units=[unit], 
                unit_idx=[i_unit], 
                tokenizer=self.tokenizer,
            )
            perturbations.append(perturbation)

        outputs = await self._get_multiple_completions([perturbation.input_string for perturbation in perturbations])
        
        for perturbation, output in zip(perturbations, outputs):
            for strategy in attribution_strategies:    
                attribution_scores, _ = self._get_scores(
                    perturbed_output=output, 
                    original_output=original_output, 
                    attribution_strategy=strategy,
                    ignore_output_token_location=ignore_output_token_location
                )

                if logger:
                    logger.log_attributions(
                        perturbation, 
                        attribution_scores,
                        strategy,
                        output.message.content
                    )

                    logger.log_perturbation(
                        len(perturbation.masked_units) - 1,
                        perturbation.masked_string,
                        str(perturbation_strategy),
                        original_input,
                        original_output.message.content,
                        perturbation.input_string,
                        output.message.content,
                    )
            
        if logger:
            logger.stop_experiment(num_llm_calls=len(outputs) + 1)

    async def hierarchical_perturbation(
            self, 
            original_input: str, 
            init_chunk_size: int,
            threshold: float = 0.5, 
            stride: Optional[int] = None, 
            logger: Optional[ExperimentLogger] = None, 
            perturbation_strategy: PerturbationStrategy = FixedPerturbationStrategy(),
            attribution_strategies: list[str] = ["cosine", "prob_diff"],
            perturb_word_wise: bool = False, 
            ignore_output_token_location: bool = True,
            verbose: bool = False,
        ) -> pd.Series:
        
        units, _, ids_per_unit = self._get_units(original_input, perturb_word_wise=perturb_word_wise)
        unit_count = len(units)

        if stride is None:
            stride = init_chunk_size

        padding = stride // 2

        # TODO: Abstract to create_masks function/method
        masks = []
        for start in range(-padding, unit_count + padding, stride):
            end = min(start + init_chunk_size, unit_count)
            mask = np.zeros(unit_count, dtype=bool)
            mask[max(start, 0):min(end, unit_count)] = True
            
            if mask.any():
                masks.append(mask)

        original_output = await self._get_chat_completion(original_input)
        if logger:
            logger.start_experiment(
                original_input,
                original_output.message.content,
                perturbation_strategy,
                perturb_word_wise,
            )

        cumulative_unit_attribution = np.zeros(unit_count)
        total_llm_calls = 1
        stage = 0

        while masks:
            print(f"Stage {stage}: making {len(masks)} perturbations")
            new_masks = []
            perturbations: list[PerturbedLLMInput] = []
            for mask in masks:
                
                perturbed_input = []
                perturbed_units = []
                for i, unit in enumerate(units):

                    # TODO: Abstract and move to PerturbedLLMInput?
                    if mask[i]:
                        perturbed_unit = [
                            self.tokenizer.decode(perturbation_strategy.get_replacement_token(token_id)).strip()
                            for token_id in ids_per_unit[i]
                        ]
                        perturbed_input.append(perturbed_unit)
                        perturbed_units.append(unit)
                    else:
                        perturbed_input.append([unit])

                perturbation = PerturbedLLMInput(
                    input_units=perturbed_input, 
                    masked_units=perturbed_units, 
                    unit_idx=np.where(mask)[0].tolist(),
                    tokenizer=self.tokenizer,
                )

                perturbations.append(perturbation)
            
            if verbose:
                print("Masked out tokens/words:")
                print(*[[perturbation.masked_string] for perturbation in perturbations], sep="\n")
            
            outputs = await self._get_multiple_completions([perturbation.input_string for perturbation in perturbations])
            total_llm_calls += len(outputs)
            
            chunk_scores = []
            unit_attribution = np.full((len(masks), unit_count), np.nan)
            
            for i, (perturbation, output, mask) in enumerate(zip(perturbations, outputs, masks)):
                
                for strategy in attribution_strategies:
                    attribution_scores, norm_attribution_scores = self._get_scores(
                        perturbed_output=output, 
                        original_output=original_output, 
                        attribution_strategy=strategy,
                        chunksize=sum(mask),
                        ignore_output_token_location=ignore_output_token_location,
                    )

                    if logger:
                        logger.log_attributions(
                            perturbation,
                            norm_attribution_scores,
                            strategy,
                            output.message.content,
                            depth=stage,
                        )

                        logger.log_perturbation(
                            len(perturbation.masked_units) - 1,
                            perturbation.masked_string,
                            str(perturbation_strategy),
                            original_input,
                            original_output.message.content,
                            perturbation.input_string,
                            output.message.content,
                        )
                
                    # For scoring we only use the first attribution strategy
                    if strategy == attribution_strategies[0]:
                        chunk_scores.append(attribution_scores["sentence_attribution"])
                        unit_attribution[i, mask] = norm_attribution_scores["sentence_attribution"]
            
            # Filling units that were not perturbed with zeros
            unperturbed_units = np.isnan(unit_attribution).all(axis=0)
            unit_attribution[:, unperturbed_units] = 0
            
            # Take mean of attribution scores and accumulate
            unit_attribution = np.nanmean(unit_attribution, axis=0)
            cumulative_unit_attribution += np.abs(unit_attribution)
            
            # Calculate midrange threshold value
            midrange_score = (np.max(cumulative_unit_attribution) + np.min(cumulative_unit_attribution)) / 2
            if midrange_score < MIN_MIDRANGE_THRESHOLD:
                break
            
            for mask, chunk_attribution in zip(masks, chunk_scores):
                cumulative_chunk_attribution = cumulative_unit_attribution[mask].mean()
                
                if (cumulative_chunk_attribution >= midrange_score or np.abs(chunk_attribution) > threshold) and mask.sum() > 1:
                    # Split the chunk in half
                    indices = np.where(mask)[0]
                    mid = len(indices) // 2
                    
                    mask1 = np.zeros(unit_count, dtype=bool)
                    mask2 = np.zeros(unit_count, dtype=bool)
                    
                    mask1[indices[:mid]] = True
                    mask2[indices[mid:]] = True
                    
                    new_masks.append(mask1)
                    new_masks.append(mask2)
            
            # Ensure masks are unique and return top-level to list
            masks = list(np.unique(new_masks, axis=0))
            stage += 1

        if logger:
            logger.df_token_attribution_matrix = logger.df_token_attribution_matrix.drop_duplicates(subset=["exp_id", "input_token_pos", "output_token"], keep="last").sort_values(by=["input_token_pos", "output_token_pos"]).reset_index(drop=True)
            logger.df_input_token_attribution = logger.df_input_token_attribution.drop_duplicates(subset=["exp_id","input_token_pos"], keep="last").sort_values(by="input_token_pos").reset_index(drop=True)
            logger.stop_experiment(num_llm_calls=total_llm_calls)

        return pd.Series(cumulative_unit_attribution, index=units, name="saliency")

    async def _get_chat_completion(self, input: str) -> StrictChoice:
        response = await self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": input}],
            temperature=0.0,
            seed=0,
            logprobs=True,
            top_logprobs=20,
        )
        return StrictChoice(**response.choices[0].model_dump())

    def _make_output_location_invariant(
            self, 
            original_output: StrictChoice, 
            perturbed_output: StrictChoice,
        ):
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

    def _get_units(self, input_text: str, perturb_word_wise: bool = False) -> tuple[list[str], list[list[str]], list[list[int]]]:
        
        # TODO: This should be abstracted, potentially moved to PerturbedLLMInput
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
                [token_id] for token_id in self.tokenizer.encode(input_text, add_special_tokens=False)
            ]

        units = ["".join(tokens) for tokens in tokens_per_unit]

        return units, tokens_per_unit, token_ids_per_unit

    def _get_scores(
            self, 
            perturbed_output: StrictChoice, 
            original_output: StrictChoice, 
            attribution_strategy: str,
            chunksize: int = 1,
            ignore_output_token_location: bool = True,
        ) -> tuple[dict[str, Any], dict[str, Any]]:

        if ignore_output_token_location:
            perturbed_output = self._make_output_location_invariant(
                original_output, perturbed_output
            )
        
        if attribution_strategy == "cosine":
            token_attributions = cosine_similarity_attribution(
                original_output.message.content,
                perturbed_output.message.content,
                self.token_embeddings,
                self.tokenizer,
            )
        elif attribution_strategy == "prob_diff":
            token_attributions = token_prob_attribution(
                original_output.logprobs, perturbed_output.logprobs
            )
        else:
            raise ValueError(f"Unknown attribution strategy: {attribution_strategy}")
        
        scores = {
            "sentence_attribution": np.mean(list(token_attributions.values())),
            "token_attribution": token_attributions
        }
        norm_scores = {
            "sentence_attribution": np.mean(list(token_attributions.values())) / chunksize,
            "token_attribution": {k: v / chunksize for k, v in token_attributions.items()}
        }

        return scores, norm_scores

    async def _get_multiple_completions(self, inputs: list[str]) -> list[StrictChoice]:

        tasks = [asyncio.create_task(self._get_chat_completion(inp)) for inp in inputs]

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
                await asyncio.sleep(REQUEST_DELAY)
        else:
            outputs = await asyncio.gather(*tasks)

        return outputs
