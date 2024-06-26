import os
import pickle
import time
from typing import Optional

import pandas as pd
from IPython.core.getipython import get_ipython

from .token_perturbation import PerturbationStrategy


class ExperimentLogger:
    def __init__(self, experiment_id=0):
        self.experiment_id = experiment_id
        self.df_experiments = pd.DataFrame(
            columns=[
                "exp_id",
                "original_input",
                "original_output",
                "perturbation_strategy",
                "perturb_word_wise",
                "duration",
            ]
        )
        self.df_input_token_attribution = pd.DataFrame(
            columns=[
                "exp_id",
                "attribution_strategy",
                "input_token_pos",
                "input_token",
                "attr_score",
            ]
        )
        self.df_token_attribution_matrix = pd.DataFrame(
            columns=[
                "exp_id",
                "attribution_strategy",
                "input_token_pos",
                "output_token_pos",
                "output_token",
                "attr_score",
                "perturbed_input",
                "perturbed_output",
            ]
        )
        self.df_perturbations = pd.DataFrame(
            columns=[
                "exp_id",
                "perturbation_pos",
                "perturbation_token",
                "perturbation_strategy",
                "original_input",
                "original_output",
                "perturbed_input",
                "perturbed_output",
            ]
        )

    def start_experiment(
        self,
        original_input: str,
        original_output: str,
        perturbation_strategy: PerturbationStrategy,
        perturb_word_wise: bool,
    ):
        self.experiment_id += 1
        self.experiment_start_time = time.time()
        self.df_experiments.loc[len(self.df_experiments)] = {
            "exp_id": self.experiment_id,
            "original_input": original_input,
            "original_output": original_output,
            "perturbation_strategy": str(perturbation_strategy),
            "perturb_word_wise": perturb_word_wise,
            "duration": None,
        }

    def stop_experiment(self):
        self.df_experiments.loc[len(self.df_experiments) - 1, "duration"] = (
            time.time() - self.experiment_start_time
        )

    def log_input_token_attribution(
        self,
        attribution_strategy: str,
        token_pos: int,
        token: str,
        attr_score: float,
    ):
        self.df_input_token_attribution.loc[len(self.df_input_token_attribution)] = {
            "exp_id": self.experiment_id,
            "attribution_strategy": attribution_strategy,
            "input_token_pos": token_pos,
            "input_token": token,
            "attr_score": attr_score,
        }

    def log_token_attribution_matrix(
        self,
        attribution_strategy: str,
        input_token_pos: int,
        output_token_pos: int,
        output_token: str,
        attr_score: float,
        perturbed_input: str,
        perturbed_output: str,
    ):
        self.df_token_attribution_matrix.loc[len(self.df_token_attribution_matrix)] = {
            "exp_id": self.experiment_id,
            "attribution_strategy": attribution_strategy,
            "input_token_pos": input_token_pos,
            "output_token_pos": output_token_pos,
            "output_token": output_token,
            "attr_score": attr_score,
            "perturbed_input": perturbed_input,
            "perturbed_output": perturbed_output,
        }

    def log_perturbation(
        self,
        perturbation_pos: int,
        perturbation_token: str,
        perturbation_strategy: str,
        original_input: str,
        original_output: str,
        perturbed_input: str,
        perturbed_output: str,
    ):
        self.df_perturbations.loc[len(self.df_perturbations)] = {
            "exp_id": self.experiment_id,
            "perturbation_pos": perturbation_pos,
            "perturbation_token": perturbation_token,
            "perturbation_strategy": perturbation_strategy,
            "original_input": original_input,
            "original_output": original_output,
            "perturbed_input": perturbed_input,
            "perturbed_output": perturbed_output,
        }

    def clean_tokens(self, tokens):
        return [token.replace("Ġ", "") for token in tokens]

    def print_sentence_attribution(self):
        sentences = []

        for (exp_id, attr_strat), exp_data in self.df_input_token_attribution.groupby(
            ["exp_id", "attribution_strategy"]
        ):
            tokens = self.clean_tokens(exp_data["input_token"].tolist())
            attr_scores = exp_data["attr_score"].tolist()

            token_attrs = [f"{token}\n{score:.2f}" for token, score in zip(tokens, attr_scores)]

            perturbation_strategy = self.df_experiments.loc[
                self.df_experiments["exp_id"] == exp_id, "perturbation_strategy"
            ].values[0]
            perturb_word_wise = self.df_experiments.loc[
                self.df_experiments["exp_id"] == exp_id, "perturb_word_wise"
            ].values[0]

            sentence_data = {
                "exp_id": exp_id,
                "attribution_strategy": attr_strat,
                "perturbation_strategy": perturbation_strategy,
                "perturb_word_wise": perturb_word_wise,
            }
            sentence_data.update(
                {f"token_{i+1}": token_attr for i, token_attr in enumerate(token_attrs)}
            )

            sentences.append(sentence_data)

        df_sentences = pd.DataFrame(sentences)
        self.pretty_print(df_sentences)

    def print_attribution_matrix(
        self,
        exp_id: int,
        attribution_strategy: Optional[str] = None,
        show_debug_cols: bool = False,
    ):
        if attribution_strategy is None:
            unique_strategies = self.df_token_attribution_matrix["attribution_strategy"].unique()
            for strategy in unique_strategies:
                self.print_attribution_matrix(
                    exp_id,
                    attribution_strategy=strategy,
                    show_debug_cols=show_debug_cols,
                )
        else:
            # Filter the data for the specific experiment and attribution strategy
            exp_data = self.df_token_attribution_matrix[
                (self.df_token_attribution_matrix["exp_id"] == exp_id)
                & (self.df_token_attribution_matrix["attribution_strategy"] == attribution_strategy)
            ]
            perturbation_strategy = self.df_experiments.loc[
                self.df_experiments["exp_id"] == exp_id, "perturbation_strategy"
            ].values[0]

            # Create the pivot table for the matrix
            matrix = exp_data.pivot(
                index="input_token_pos", columns="output_token_pos", values="attr_score"
            )

            # Retrieve and clean tokens
            input_tokens = self.clean_tokens(
                self.df_input_token_attribution[
                    (self.df_input_token_attribution["exp_id"] == exp_id)
                    & (
                        self.df_input_token_attribution["attribution_strategy"]
                        == attribution_strategy
                    )
                ]["input_token"].tolist()
            )

            output_tokens = self.clean_tokens(
                exp_data.loc[exp_data["input_token_pos"] == 0, "output_token"].tolist()
            )

            # Append positions to tokens for uniqueness
            input_tokens_with_pos = [f"{token} ({i})" for i, token in enumerate(input_tokens)]
            output_tokens_with_pos = [f"{token} ({i})" for i, token in enumerate(output_tokens)]

            # Retrieve the output tokens for the columns
            output_tokens = exp_data["output_token"].unique().tolist()

            # Set the row and column names of the matrix
            matrix.index = input_tokens_with_pos
            matrix.columns = output_tokens_with_pos

            print(
                f"Attribution matrix for experiment {exp_id} \nAttribution Strategy: {attribution_strategy} \nPerturbation strategy: {perturbation_strategy}:"
            )
            print("Input Tokens (Rows) vs. Output Tokens (Columns)")

            if show_debug_cols:
                additional_columns = exp_data[
                    ["input_token_pos", "perturbed_input", "perturbed_output"]
                ].drop_duplicates()
                additional_columns = additional_columns.set_index(
                    additional_columns["input_token_pos"].apply(
                        lambda x: f"{input_tokens[x]} ({x})"
                    )
                )
                additional_columns = additional_columns[["perturbed_input", "perturbed_output"]]
                matrix = matrix.join(additional_columns)
            if "IPKernelApp" in get_ipython().config:
                from IPython.display import display

                display(matrix.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1))
            else:
                print(matrix)

    def pretty_print(self, df: pd.DataFrame):
        # Check if code is running in Jupyter notebook
        if "IPKernelApp" in get_ipython().config:
            from IPython.display import display

            display(df.style.set_properties(**{"white-space": "pre-wrap"}))
        else:
            print(df.to_string())

    def print_tables(self):
        print("Message Table:")
        self.pretty_print(self.df_experiments)
        print("\nAttribution Table:")
        self.pretty_print(self.format_attribution_table())
        print("\nPerturbed Message Table:")
        self.pretty_print(self.df_perturbations)

    def format_attribution_table(self):
        df_pivot = self.df_token_attr.pivot(
            index="exp_id", columns="token_pos", values=["token", "attr_score"]
        )
        # Concatenate token and attr_score into a single string in each cell
        for pos in range(
            df_pivot["token"].shape[1]
        ):  # Assuming the number of positions is the number of columns in 'token'
            df_pivot["attr_score", pos] = pd.to_numeric(
                df_pivot["attr_score", pos], errors="coerce"
            )
            df_pivot["attr_score", pos] = df_pivot["attr_score", pos].round(2)
            df_pivot[f"token_{pos}"] = (
                df_pivot["token", pos].astype(str) + "\n" + df_pivot["attr_score", pos].astype(str)
            )

        df_pivot = df_pivot.loc[:, df_pivot.columns.get_level_values(0).str.startswith("token_")]
        df_pivot.reset_index(inplace=True)
        styled_df = df_pivot.style.set_properties(**{"white-space": "pre-wrap"})
        return styled_df

    def save(self, path):
        with open(os.path.join(path, "experiment_logger.pkl"), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            loaded_logger = pickle.load(f)

        loaded_logger.experiment_id = loaded_logger.df_experiments["exp_id"].max() + 1
        return loaded_logger
