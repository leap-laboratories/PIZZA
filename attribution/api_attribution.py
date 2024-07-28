import asyncio
import os
from copy import deepcopy
from typing import Any, Literal, Optional, cast

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
    LLMInput,
    PerturbationStrategy,
    PerturbedLLMInput,
    get_masks,
    split_mask,
)
from .types import StrictChoice

load_dotenv()

DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
REQUEST_DELAY = 0.1
MIN_MAXIMUM_THRESHOLD = 0.01
CHUNK_DIVISIOR = 2


class OpenAIAttributor(BaseAsyncLLMAttributor):
    def __init__(
        self,
        openai_model: str = DEFAULT_OPENAI_MODEL,
        openai_api_key: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        token_embeddings: Optional[np.ndarray] = None,
        max_concurrent_requests: Optional[int] = 50,
    ):
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.openai_model = openai_model

        self.tokenizer = tokenizer or GPT2Tokenizer.from_pretrained("gpt2")
        self.token_embeddings = (
            token_embeddings
            or cast(GPT2LMHeadModel, GPT2LMHeadModel.from_pretrained("gpt2"))
            .transformer.wte.weight.detach()
            .numpy()
        )
        self.max_concurrent_requests = max_concurrent_requests

    async def compute_attributions(
        self,
        original_input: str,
        perturbation_strategy: PerturbationStrategy = FixedPerturbationStrategy(),
        attribution_strategies: list[str] = ["prob_diff"],
        logger: Optional[ExperimentLogger] = None,
        unit_definition: Literal["token", "word"] = "token",
        ignore_output_token_location: bool = True,
    ):
        llm_input = LLMInput(
            input_string=original_input, tokenizer=self.tokenizer, unit_definition=unit_definition
        )
        original_output = await self.get_chat_completion(llm_input.input_string)

        if logger:
            logger.start_experiment(
                original_input,
                original_output.message.content,
                perturbation_strategy,
                unit_definition,
            )

        perturbations: list[PerturbedLLMInput] = []
        for i_unit in range(len(llm_input.unit_tokens)):
            perturbation = PerturbedLLMInput(
                original=llm_input,
                perturb_unit_ids=[i_unit],
                strategy=perturbation_strategy,
            )
            perturbations.append(perturbation)

        outputs = await self.get_multiple_completions(
            [perturbation.perturbed_string for perturbation in perturbations]
        )

        for perturbation, output in zip(perturbations, outputs):
            for strategy in attribution_strategies:
                _ = self._get_scores(
                    perturbation=perturbation,
                    perturbed_output=output,
                    original_output=original_output,
                    attribution_strategy=strategy,
                    ignore_output_token_location=ignore_output_token_location,
                    logger=logger,
                )

        if logger:
            logger.stop_experiment(num_llm_calls=len(outputs) + 1)

    async def hierarchical_perturbation(
        self,
        original_input: str,
        init_chunk_size: Optional[int] = None,
        stride: Optional[int] = None,
        perturbation_strategy: PerturbationStrategy = FixedPerturbationStrategy(),
        attribution_strategies: list[str] = ["prob_diff"],
        static_threshold: Optional[float] = None,
        use_absolute_attribution: bool = False,
        unit_definition: Literal["token", "word"] = "token",
        ignore_output_token_location: bool = True,
        logger: Optional[ExperimentLogger] = None,
        verbose: int = 0,
    ) -> None:
        """
        Hierarchical pertubation method. Uses a sliding window to split the input into chunks and continues to subdivided each chunk until the attribution falls below the dynamic threshold.
        Args:
            input_prompt (str): The original input string.
            init_chunk_size (int): The initial chunk size for splitting the input.
            stride (Optional[int]): The stride for sliding the window. Defaults to None.
            perturbation_strategy (PerturbationStrategy): The perturbation strategy to use. Defaults to FixedPerturbationStrategy().
            attribution_strategies (list[str]): The list of attribution strategies to use. Defaults to ["cosine", "prob_diff"].
            static_threshold (Optional[float]): The static threshold for chunk attribution scores at each depth. Defaults to None.
            use_absolute_attribution (bool): Flag indicating whether to use absolute attribution scores in dynamic threshold calculation. Defaults to False.
            unit_definition (Literal["token", "word"]): The unit definition for splitting the input. Defaults to "token".
            ignore_output_token_location (bool): Flag indicating whether to ignore the output token location. Defaults to True.
            logger (Optional[ExperimentLogger]): The experiment logger. Defaults to None.
            verbose (bool): Flag indicating whether to print verbose output. Defaults to False.
        """
        llm_input = LLMInput(
            input_string=original_input, tokenizer=self.tokenizer, unit_definition=unit_definition
        )

        if init_chunk_size is None:
            init_chunk_size = max(2, len(llm_input.unit_tokens) // CHUNK_DIVISIOR)
        if stride is None:
            stride = max(1, init_chunk_size//2)

        original_output = await self.get_chat_completion(llm_input.input_string)
        if logger:
            logger.start_experiment(
                original_input,
                original_output.message.content,
                perturbation_strategy,
                unit_definition,
            )

        # Defining boolean masks via a sliding window to split the input into chunks
        unit_count = len(llm_input.unit_tokens)
        masks = get_masks(unit_count, init_chunk_size, stride)

        comparator = np.abs if use_absolute_attribution else lambda x: x
        cumulative_unit_attribution = np.zeros(unit_count)
        total_llm_calls = 1
        stage = 0

        # Main loop for hierarchical perturbation, run until masks cannot be further subdivided
        while masks:
            if verbose > 0:
                print(f"Stage {stage}: making {len(masks)} perturbations")
            # Define perturbations for each mask
            perturbations: list[PerturbedLLMInput] = []
            for mask in masks:
                perturbations.append(
                    PerturbedLLMInput(
                        original=llm_input,
                        strategy=perturbation_strategy,
                        perturb_unit_ids=np.where(mask)[0].tolist(),
                    )
                )

            if verbose > 1:
                print("Masked out tokens/words:")
                print(*[[perturbation.masked_string] for perturbation in perturbations], sep=" ")

            # Wait for the perturbed results
            outputs = await self.get_multiple_completions(
                [perturbation.perturbed_string for perturbation in perturbations]
            )
            total_llm_calls += len(outputs)

            # Calculate attribution scores for each perturbation
            chunk_scores = []
            unit_attribution = np.full((len(masks), unit_count), np.nan)

            for i, (perturbation, output, mask) in enumerate(zip(perturbations, outputs, masks)):
                for strategy in attribution_strategies:
                    # Logging each attribution strategy metric
                    attribution_scores, norm_attribution_scores = self._get_scores(
                        perturbation=perturbation,
                        perturbed_output=output,
                        original_output=original_output,
                        attribution_strategy=strategy,
                        chunksize=sum(mask),
                        ignore_output_token_location=ignore_output_token_location,
                        logger=logger,
                        depth=stage,
                    )

                    # For scoring we only use the first attribution strategy
                    if strategy == attribution_strategies[0]:
                        chunk_scores.append(attribution_scores["total_attribution"])
                        unit_attribution[i, mask] = norm_attribution_scores["total_attribution"]

            # Filling units that were not perturbed with zeros to avoid full nan columns
            unperturbed_units = np.isnan(unit_attribution).all(axis=0)
            unit_attribution[:, unperturbed_units] = 0

            # Take mean of attribution scores and accumulate
            unit_attribution = np.nanmean(unit_attribution, axis=0)
            cumulative_unit_attribution += comparator(unit_attribution)

            if np.max(cumulative_unit_attribution) > MIN_MAXIMUM_THRESHOLD:
                # Calculate midrange threshold value
                midrange_score = (
                    np.max(cumulative_unit_attribution) + np.min(cumulative_unit_attribution)
                ) / 2
            else:
                break

            # Split masks if their scores exceed the midrange score or a static threshold
            new_masks = []
            for mask, chunk_attribution in zip(masks, chunk_scores):
                cumulative_chunk_attribution = cumulative_unit_attribution[mask].mean()

                if (
                    cumulative_chunk_attribution >= midrange_score
                    or (
                        static_threshold is not None
                        and comparator(chunk_attribution) > static_threshold
                    )
                ) and mask.sum() > 1:
                    # Split the chunk in half
                    mask1, mask2 = split_mask(mask)
                    new_masks.append(mask1)
                    new_masks.append(mask2)

            # Ensure masks are unique and return top-level to list
            masks = list(np.unique(new_masks, axis=0))
            stage += 1

        if logger:
            logger.stop_experiment(num_llm_calls=total_llm_calls)

    async def get_chat_completion(self, input: str) -> StrictChoice:
        response = await self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": input}],
            temperature=0.0,
            seed=0,
            logprobs=True,
            top_logprobs=20,
        )
        return StrictChoice(**response.choices[0].model_dump())

    async def get_multiple_completions(self, inputs: list[str]) -> list[StrictChoice]:
        tasks = [asyncio.create_task(self.get_chat_completion(inp)) for inp in inputs]

        # Get the output logprobs for the perturbed inputs
        if self.max_concurrent_requests is not None and len(tasks) > self.max_concurrent_requests:
            outputs = []
            for idx in tqdm(
                range(0, len(tasks), self.max_concurrent_requests),
                desc=f"Sending {self.max_concurrent_requests:.0f} concurrent requests at a time",
            ):
                batch = [
                    tasks[i]
                    for i in range(idx, min(idx + self.max_concurrent_requests, len(tasks)))
                ]
                outputs.extend(await asyncio.gather(*batch))
                await asyncio.sleep(REQUEST_DELAY)
        else:
            outputs = await asyncio.gather(*tasks)

        return outputs

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

    def _get_scores(
        self,
        perturbation: PerturbedLLMInput,
        perturbed_output: StrictChoice,
        original_output: StrictChoice,
        attribution_strategy: str,
        chunksize: int = 1,
        ignore_output_token_location: bool = True,
        depth: int = 0,
        logger: Optional[ExperimentLogger] = None,
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
            "total_attribution": np.mean(list(token_attributions.values())),
            "token_attribution": token_attributions,
        }
        norm_scores = {
            "total_attribution": np.mean(list(token_attributions.values())) / chunksize,
            "token_attribution": {k: v / chunksize for k, v in token_attributions.items()},
        }

        if logger:
            logger.log_attributions(
                perturbation=perturbation,
                attribution_scores=norm_scores,
                strategy=attribution_strategy,
                output=perturbed_output.message.content,
                depth=depth,
            )

            logger.log_perturbation(
                perturbation_pos=len(perturbation.masked_units) - 1,
                perturbation_token=perturbation.masked_string,
                perturbation_strategy=str(attribution_strategy),
                original_input=perturbation.original.input_string,
                original_output=original_output.message.content,
                perturbed_input=perturbation.perturbed_string,
                perturbed_output=perturbed_output.message.content,
            )

        return scores, norm_scores
