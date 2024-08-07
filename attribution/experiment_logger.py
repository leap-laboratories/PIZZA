import os
import pickle
import time
from typing import Any, Literal, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.getipython import get_ipython
from IPython.display import HTML

from .token_perturbation import PerturbationStrategy, PerturbedLLMInput, combine_unit

DIV_STYLE_STR = '<div style="font-family: monospace; line-height: 1.5;">'
SPAN_STYLE_COLOR_STR = '<span style="text-decoration: underline; text-decoration-color: #color#; text-decoration-thickness: 4px; text-underline-offset: 3px;">'
SPAN_STYLE_STR = '<span style="text-decoration: underline; text-decoration-thickness: 4px; text-underline-offset: 3px;">'


class ExperimentLogger:
    def __init__(self, experiment_id=0):
        self.experiment_id = experiment_id
        self.df_experiments = pd.DataFrame(
            columns=[
                "exp_id",
                "original_input",
                "original_output",
                "perturbation_strategy",
                "unit_definition",
                "duration",
                "num_llm_calls",
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
                "depth",
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
        unit_definition: Literal["token", "word"],
    ):
        self.experiment_id += 1
        self.experiment_start_time = time.time()
        self.df_experiments.loc[len(self.df_experiments)] = {
            "exp_id": self.experiment_id,
            "original_input": original_input,
            "original_output": original_output,
            "perturbation_strategy": str(perturbation_strategy),
            "unit_definition": unit_definition,
            "duration": None,
            "num_llm_calls": None,
        }

    def stop_experiment(self, num_llm_calls: Optional[int] = None):
        self.df_experiments.loc[len(self.df_experiments) - 1, "duration"] = (
            time.time() - self.experiment_start_time
        )
        self.df_experiments.loc[len(self.df_experiments) - 1, "num_llm_calls"] = num_llm_calls

    def log_attributions(
        self,
        perturbation: PerturbedLLMInput,
        attribution_scores: dict[str, Any],
        strategy: str,
        output: str,
        depth: int = 0,
    ):
        for unit_token, token_id in zip(perturbation.masked_units, perturbation.perturb_unit_ids):
            self.log_input_token_attribution(
                strategy,
                token_id,
                combine_unit(unit_token),
                float(attribution_scores["total_attribution"]),
            )

            for j, (output_token, attr_score) in enumerate(
                attribution_scores["token_attribution"].items()
            ):
                self.log_token_attribution_matrix(
                    strategy,
                    token_id,
                    j,
                    " ".join(output_token.split(" ")[:-1]),
                    attr_score,
                    perturbation.perturbed_string,
                    output,
                    depth,
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
        depth: int = 0,
    ):
        self.df_token_attribution_matrix.loc[len(self.df_token_attribution_matrix)] = {
            "exp_id": self.experiment_id,
            "attribution_strategy": attribution_strategy,
            "input_token_pos": input_token_pos,
            "output_token_pos": output_token_pos,
            "output_token": output_token,
            "attr_score": attr_score,
            "depth": depth,
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
        return [token.replace("Ġ", " ") for token in tokens]

    def score_to_color(self, score, vmin=-1, vmax=1):
        # Setting vmin and vmax to -1 and 1 centers the scores around 0.
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.seismic
        rgba_color = cmap(norm(score))
        color_hex = mcolors.to_hex(rgba_color)
        return color_hex

    def print_text_total_attribution(
        self, exp_id: Optional[int] = None, score_agg: Literal["mean", "sum", "last"] = "mean"
    ):
        if exp_id is not None and exp_id < 0:
            exp_id = self.df_experiments["exp_id"].max() + 1 + exp_id

        token_attrs_df = (
            self.df_input_token_attribution.groupby(["exp_id", "attribution_strategy"])
            if exp_id is None
            else self.df_input_token_attribution[
                self.df_input_token_attribution["exp_id"] == exp_id
            ].groupby(["exp_id", "attribution_strategy"])
        )

        for (exp_id, _), exp_data in token_attrs_df:
            if exp_data["input_token_pos"].duplicated().any():
                exp_data = self._aggregate_attr_score_df(exp_data, score_agg)

            tokens = self.clean_tokens(exp_data["input_token"].tolist())
            attr_scores = exp_data["attr_score"].tolist()

            token_dict = {f"token_{i+1}": t for i, t in enumerate(tokens)}
            score_dict = {f"token_{i+1}": score for i, score in enumerate(attr_scores)}

            output = self.df_experiments.loc[
                self.df_experiments["exp_id"] == exp_id, "original_output"
            ].values[0]

            df = pd.DataFrame([token_dict, score_dict], index=["token", "attr_score"])

            # Generating HTML
            html_str = DIV_STYLE_STR
            for col in df.columns:
                token = df[col]["token"]
                score = df[col]["attr_score"]
                color = self.score_to_color(score)
                html_str += SPAN_STYLE_COLOR_STR.replace("#color#", color) + token + "</span>"

            html_str += " -> " + output
            html_str += "</div>"

            # Display

            if get_ipython() and "IPKernelApp" in get_ipython().config:
                from IPython.display import display

                display(HTML(html_str))
            else:
                self.pretty_print(df)

    def print_text_attribution_matrix(self, exp_id: int = -1):
        if exp_id is not None and exp_id < 0:
            exp_id = self.df_experiments["exp_id"].max() + 1 + exp_id

        matrices = self.get_attribution_matrices(exp_id)

        for matrix in matrices:
            input_tokens = [" ".join(x.split(" ")[:-1]) for x in matrix.index]

            token_dict = {f"token_{i+1}": t for i, t in enumerate(input_tokens)}

            for oi, output_token in enumerate(matrix.columns):
                prev_output_str = "".join(
                    [" ".join(ot.split(" ")[:-1]) for ot in matrix.columns[:oi]]
                )
                following_output_str = "".join(
                    [" ".join(ot.split(" ")[:-1]) for ot in matrix.columns[oi + 1 :]]
                )
                attr_scores = matrix[output_token].tolist()

                score_dict = {f"token_{i+1}": score for i, score in enumerate(attr_scores)}

                df = pd.DataFrame([token_dict, score_dict], index=["token", "attr_score"])

                # Generating HTML
                html_str = DIV_STYLE_STR
                for col in df.columns:
                    token = df[col]["token"]
                    score = df[col]["attr_score"]
                    color = self.score_to_color(score)
                    html_str += SPAN_STYLE_COLOR_STR.replace("#color#", color) + token + "</span>"

                clean_output_token = " ".join(output_token.split(" ")[:-1])
                html_str += (
                    " -> "
                    + prev_output_str
                    + SPAN_STYLE_STR
                    + clean_output_token
                    + "</span>"
                    + following_output_str
                )
                html_str += "</div>"

                # Display

                if get_ipython() and "IPKernelApp" in get_ipython().config:
                    from IPython.display import display

                    display(HTML(html_str))
                else:
                    self.pretty_print(df)

    def print_total_attribution(
        self, exp_id: Optional[int] = None, score_agg: Literal["mean", "last"] = "mean"
    ):
        totals = []
        if exp_id is not None and exp_id < 0:
            exp_id = self.df_experiments["exp_id"].max() + 1 + exp_id

        token_attrs_df = (
            self.df_input_token_attribution.groupby(["exp_id", "attribution_strategy"])
            if exp_id is None
            else self.df_input_token_attribution[
                self.df_input_token_attribution["exp_id"] == exp_id
            ].groupby(["exp_id", "attribution_strategy"])
        )

        for (exp_id, attr_strategy), exp_data in token_attrs_df:
            if exp_data["input_token_pos"].duplicated().any():
                exp_data = self._aggregate_attr_score_df(exp_data, score_agg)

            tokens = self.clean_tokens(exp_data["input_token"].tolist())
            attr_scores = exp_data["attr_score"].tolist()

            token_attrs = [f"{token}\n{score:.2f}" for token, score in zip(tokens, attr_scores)]

            perturbation_strategy = self.df_experiments.loc[
                self.df_experiments["exp_id"] == exp_id, "perturbation_strategy"
            ].values[0]
            unit_definition = self.df_experiments.loc[
                self.df_experiments["exp_id"] == exp_id, "unit_definition"
            ].values[0]

            total_data = {
                "exp_id": exp_id,
                "attribution_strategy": attr_strategy,
                "perturbation_strategy": perturbation_strategy,
                "unit_definition": unit_definition,
            }
            total_data.update(
                {f"token_{i+1}": token_attr for i, token_attr in enumerate(token_attrs)}
            )
            totals.append(total_data)

        df_totals = pd.DataFrame(totals)
        self.pretty_print(df_totals)
        return df_totals

    def print_attribution_matrix(
        self,
        exp_id: int = -1,
        attribution_strategy: Optional[str] = None,
        show_debug_cols: bool = False,
        score_agg: Literal["mean", "last"] = "mean",
    ):
        if exp_id is not None and exp_id < 0:
            exp_id = self.df_experiments["exp_id"].max() + 1 + exp_id

        matrices = self.get_attribution_matrices(
            exp_id, attribution_strategy, show_debug_cols, score_agg
        )

        for matrix in matrices:
            if get_ipython() and "IPKernelApp" in get_ipython().config:
                from IPython.display import display

                display(
                    matrix.style.background_gradient(
                        cmap="seismic", vmin=-1, vmax=1
                    ).set_properties(**{"white-space": "pre-wrap"})
                )
            else:
                self.pretty_print(matrix)

    def get_attribution_matrices(
        self,
        exp_id: int = -1,
        attribution_strategy: Optional[str] = None,
        show_debug_cols: bool = False,
        score_agg: Literal["mean", "sum", "last"] = "mean",
    ):
        if exp_id is not None and exp_id < 0:
            exp_id = self.df_experiments["exp_id"].max() + 1 + exp_id

        if attribution_strategy is None:
            strategies = self.df_token_attribution_matrix[
                (self.df_token_attribution_matrix["exp_id"] == exp_id)
            ]["attribution_strategy"].unique()
        else:
            strategies = [attribution_strategy]

        matrices = []
        for attribution_strategy in strategies:
            # Filter the data for the specific experiment and attribution strategy
            exp_data = self.df_token_attribution_matrix[
                (self.df_token_attribution_matrix["exp_id"] == exp_id)
                & (self.df_token_attribution_matrix["attribution_strategy"] == attribution_strategy)
            ]

            token_data = self.df_input_token_attribution[
                (self.df_input_token_attribution["exp_id"] == exp_id)
                & (self.df_input_token_attribution["attribution_strategy"] == attribution_strategy)
            ]

            is_hierarchical = exp_data["depth"].any()

            if is_hierarchical:
                exp_data = self._aggregate_attr_score_df(exp_data, score_agg)
                token_data = self._aggregate_attr_score_df(token_data, score_agg)

            # Create the pivot table for the matrix

            matrix = exp_data.pivot(
                index="input_token_pos", columns="output_token_pos", values="attr_score"
            )

            # Retrieve and clean tokens
            input_tokens = self.clean_tokens(token_data["input_token"].tolist())

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

            matrices.append(matrix)
        return matrices

    def pretty_print(self, df: pd.DataFrame):
        # Check if code is running in Jupyter notebook
        if get_ipython() and "IPKernelApp" in get_ipython().config:
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

    def format_attribution_table(self) -> pd.DataFrame:
        df_pivot = self.df_input_token_attribution.pivot(
            index="exp_id", columns="input_token_pos", values=["input_token", "attr_score"]
        )
        # Concatenate token and attr_score into a single string in each cell
        for pos in range(
            df_pivot["input_token"].shape[1]
        ):  # Assuming the number of positions is the number of columns in 'token'
            df_pivot["attr_score", pos] = pd.to_numeric(
                df_pivot["attr_score", pos], errors="coerce"
            )
            df_pivot["attr_score", pos] = df_pivot["attr_score", pos].round(2)
            df_pivot[f"input_token_{pos}"] = (
                df_pivot["input_token", pos].astype(str)
                + "\n"
                + df_pivot["attr_score", pos].astype(str)
            )

        df_pivot = df_pivot.loc[
            :, df_pivot.columns.get_level_values(0).str.startswith("input_token_")
        ]
        df_pivot.reset_index(inplace=True)
        return df_pivot

    def _aggregate_attr_score_df(
        self, df: pd.DataFrame, score_agg: Literal["mean", "sum", "last"]
    ) -> pd.DataFrame:
        """
        Aggregate duplicate perturbed tokens, only relevant for hierarchical perturbation methods
        df: DataFrame containing the token attribution scores (generally one of the attribution tables)
        score_agg: Method to aggregate the scores, either "mean" across all depths, "sum" which produces a saliency map, or "last" which gives scores similar to the non-hierarchical method
        """

        aggregation_dict = dict.fromkeys(df.columns, "last")
        aggregation_dict.update({"attr_score": score_agg})
        aggregation_dict.pop("input_token_pos")

        groupby_columns = ["input_token_pos"]
        if "output_token_pos" in df.columns:
            aggregation_dict.pop("output_token_pos")
            groupby_columns.append("output_token_pos")

        df = (
            df.groupby(groupby_columns)
            .agg(aggregation_dict)
            .sort_values(by=groupby_columns)
            .reset_index()
        )

        return df

    def save(self, path):
        with open(os.path.join(path, "experiment_logger.pkl"), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            loaded_logger = pickle.load(f)

        loaded_logger.experiment_id = loaded_logger.df_experiments["exp_id"].max() + 1
        return loaded_logger
