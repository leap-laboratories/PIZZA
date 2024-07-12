import asyncio
import os
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

    def get_units(self, input_text, **kwargs) -> tuple[list[str], list[list[str]], list[list[int]]]:

        perturb_word_wise: bool = kwargs.get("perturb_word_wise", False)

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
    
    def get_perturbation(self, perturbed_input: list[list[str]], perturbed_units: list[str], perturbed_idx: list[int]):
        
        return {
            "input_tokens": perturbed_input,
            "input_string": self.tokenizer.convert_tokens_to_string(["".join(unit) for unit in perturbed_input]),
            "masked_tokens": perturbed_units,
            "masked_string": self.tokenizer.convert_tokens_to_string(perturbed_units).strip(),
            "token_idx": perturbed_idx,
        }

    async def hierarchical_perturbation(self, original_input: str, init_chunksize: int, threshold: float = 0.5, perturb_word_wise: bool = False, logger: Optional[ExperimentLogger] = None, **kwargs):
        
        units, _, ids_per_unit = self.get_units(original_input, perturb_word_wise=perturb_word_wise)
        unit_count = len(units)

        perturbation_strategy: PerturbationStrategy = kwargs.get(
            "perturbation_strategy", FixedPerturbationStrategy()
        )

        attribution_strategies: List[str] = kwargs.get(
            "attribution_strategies", ["cosine", "prob_diff"]
        )

        # Initialize mask with initial chunks
        masks = []
        for start in range(0, unit_count, init_chunksize):
            end = min(start + init_chunksize, unit_count)
            mask = np.zeros(unit_count, dtype=bool)
            mask[start:end] = True
            masks.append(mask)

        original_output = await self.get_chat_completion(original_input)
        if logger:
            logger.start_experiment(
                original_input,
                original_output.message.content,
                perturbation_strategy,
                perturb_word_wise,
            )

        cumulative_unit_scores = np.zeros(unit_count)
        total_llm_calls = 1
        stage = 0

        while masks:
            print(f"Stage {stage}")
            print("Masked out tokens/words:")
            new_masks = []
            perturbations = []
            for mask in masks:
                
                perturbed_input = []
                perturbed_units = []
                for i, unit in enumerate(units):
                    if mask[i]:
                        perturbed_unit = [
                            self.tokenizer.decode(perturbation_strategy.get_replacement_token(token_id)).strip()
                            for token_id in ids_per_unit[i]
                        ]
                        perturbed_input.append(perturbed_unit)
                        perturbed_units.append(unit)
                    else:
                        perturbed_input.append(unit)

                perturbation = self.get_perturbation(
                    perturbed_input, 
                    perturbed_units, 
                    np.where(mask)[0].tolist()
                )
                print([perturbation["masked_string"]])

                perturbations.append(perturbation)
            
            outputs = await self.compute_attribution_chunks([perturbation["input_string"] for perturbation in perturbations])
            total_llm_calls += len(outputs)
            
            perturbation_scores = []
            unit_scores = np.zeros(unit_count)
            
            for perturbation, output, mask in zip(perturbations, outputs, masks):
                attribution_scores, norm_attribution_scores = self.get_scores(output, original_output, sum(mask), **kwargs)
                perturbation_scores.append(attribution_scores[attribution_strategies[0]]["sentence_attribution"])
                unit_scores[mask] = norm_attribution_scores[attribution_strategies[0]]["sentence_attribution"]
            
                if logger:
                    logger.log_attributions(
                        perturbation,
                        norm_attribution_scores,
                        output.message.content,
                    )

                    logger.log_perturbation(
                        len(perturbation["masked_tokens"]) - 1,
                        # TODO: Should come from pertubations, could be different tokens
                        perturbation_strategy.replacement_token,
                        str(perturbation_strategy),
                        original_input,
                        original_output.message.content,
                        perturbation["input_string"],
                        output.message.content,
                    )
            
            cumulative_unit_scores += unit_scores

            midrange_score = (np.max(perturbation_scores) + np.min(perturbation_scores)) / 2
            if midrange_score < MIN_MIDRANGE_THRESHOLD:
                break
            
            for mask, score in zip(masks, perturbation_scores):
                if (score >= midrange_score or score > threshold) and mask.sum() > 1:
                    # Split the chunk in half
                    indices = np.where(mask)[0]
                    mid = len(indices) // 2
                    
                    mask1 = np.zeros(unit_count, dtype=bool)
                    mask2 = np.zeros(unit_count, dtype=bool)
                    
                    mask1[indices[:mid]] = True
                    mask2[indices[mid:]] = True
                    
                    new_masks.append(mask1)
                    new_masks.append(mask2)

            masks = new_masks
            stage += 1

        if logger:
            logger.df_token_attribution_matrix = logger.df_token_attribution_matrix.drop_duplicates(subset=["exp_id", "input_token_pos", "output_token"], keep="last").sort_values(by=["input_token_pos", "output_token_pos"]).reset_index(drop=True)
            logger.df_input_token_attribution = logger.df_input_token_attribution.drop_duplicates(subset=["exp_id","input_token_pos"], keep="last").sort_values(by="input_token_pos").reset_index(drop=True)
            logger.stop_experiment(num_llm_calls=total_llm_calls)

        return cumulative_unit_scores.tolist()

    def get_scores(self, perturbed_output, original_output, chunksize: int, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        attribution_strategies: List[str] = kwargs.get(
            "attribution_strategies", ["cosine", "prob_diff"]
        )

        ignore_output_token_location: bool = kwargs.get("ignore_output_token_location", True)

        if ignore_output_token_location:
            output = self.make_output_location_invariant(
                original_output, perturbed_output
            )
        
        scores = {}
        norm_scores = {}
        for attribution_strategy in attribution_strategies:
            if attribution_strategy == "cosine":
                token_attributions = (
                    cosine_similarity_attribution(
                        original_output.message.content,
                        output.message.content,
                        self.token_embeddings,
                        self.tokenizer,
                    )
                )
            elif attribution_strategy == "prob_diff":
                token_attributions = token_prob_attribution(
                    original_output.logprobs, output.logprobs
                )
            else:
                raise ValueError(f"Unknown attribution strategy: {attribution_strategy}")
            
            scores[attribution_strategy] = {
                "sentence_attribution": np.mean(list(token_attributions.values())),
                "token_attribution": token_attributions
            }            
            norm_scores[attribution_strategy] = {
                "sentence_attribution": np.mean(list(token_attributions.values())) / chunksize,
                "token_attribution": {k: v / chunksize for k, v in token_attributions.items()}
            }

        return scores, norm_scores

    async def compute_attribution_chunks(self, inputs: list[str]) -> list[openai.types.CompletionChoice]:

        tasks = [asyncio.create_task(self.get_chat_completion(inp)) for inp in inputs]

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

    async def compute_attributions(self, original_input: str, **kwargs):
        perturbation_strategy: PerturbationStrategy = kwargs.get(
            "perturbation_strategy", FixedPerturbationStrategy()
        )

        logger: ExperimentLogger = kwargs.get("logger", None)
        perturb_word_wise: bool = kwargs.get("perturb_word_wise", False)

        original_output = await self.get_chat_completion(original_input)

        if logger:
            logger.start_experiment(
                original_input,
                original_output.message.content,
                perturbation_strategy,
                perturb_word_wise,
            )

        units, tokens_per_unit, ids_per_unit = self.get_units(original_input, perturb_word_wise=perturb_word_wise)

        perturbations = []
        for i_unit, unit in enumerate(units):
            perturbed_unit = [
                self.tokenizer.decode(perturbation_strategy.get_replacement_token(token_id)).strip()
                for token_id in ids_per_unit[i_unit]
            ]

            perturbed_input = deepcopy(tokens_per_unit)
            perturbed_input[i_unit] = perturbed_unit

            perturbation = self.get_perturbation(perturbed_input, [unit], [i_unit])
            perturbations.append(perturbation)

        outputs = await self.compute_attribution_chunks([perturbation["input_string"] for perturbation in perturbations])
        
        for perturbation, output in zip(perturbations, outputs):
            attribution_scores, _ = self.get_scores(output, original_output, 1, **kwargs)
            
            if logger:
                logger.log_attributions(
                    perturbation, 
                    attribution_scores, 
                    output.message.content
                )

                logger.log_perturbation(
                    len(perturbation["masked_tokens"]) - 1,
                    # TODO: Should come from pertubations, could be different tokens
                    perturbation_strategy.replacement_token,
                    str(perturbation_strategy),
                    original_input,
                    original_output.message.content,
                    perturbation["input_string"],
                    output.message.content,
                )
            
        if logger:
            logger.stop_experiment(num_llm_calls=len(outputs) + 1)
