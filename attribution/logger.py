import os
import pickle

import pandas as pd
from IPython.core.getipython import get_ipython


class ExperimentLogger:
    def __init__(self, experiment_id=0):
        self.experiment_id = experiment_id
        self.df_experiments = pd.DataFrame(
            columns=[
                "exp_id",
                "input",
                "output",
                "perturbation_strategy",
                "attribution_strategy",
            ]
        )
        self.df_token_attr = pd.DataFrame(
            columns=["exp_id", "token_pos", "token", "attr_score"]
        )
        self.df_perturbations = pd.DataFrame(
            columns=[
                "exp_id",
                "input",
                "perturbed_input",
                "perturbation_pos",
                "perturbation_token",
                "attr_score",
            ]
        )

    def log_experiment(
        self,
        input_message: str,
        output_message: str,
        perturbation_strategy: str,
        attribution_strategy: str,
    ):
        self.experiment_id += 1
        self.df_experiments.loc[len(self.df_experiments)] = {
            "exp_id": self.experiment_id,
            "input": input_message,
            "output": output_message,
            "perturbation_strategy": perturbation_strategy,
            "attribution_strategy": attribution_strategy,
        }

    def log_token_attr(self, token_pos: int, token: str, attr_score: float):
        self.df_token_attr.loc[len(self.df_token_attr)] = {
            "exp_id": self.experiment_id,
            "token_pos": token_pos,
            "token": token,
            "attr_score": attr_score,
        }

    def log_perturbation(
        self,
        message: str,
        perturbed_message: str,
        perturbation_pos: int,
        token_perturbation: str,
        attr_score: float,
    ):
        self.df_perturbations.loc[len(self.df_perturbations)] = {
            "exp_id": self.experiment_id,
            "input": message,
            "perturbed_input": perturbed_message,
            "perturbation_pos": perturbation_pos,
            "perturbation_token": token_perturbation,
            "attr_score": attr_score,
        }

    def save(self, path):
        with open(os.path.join(path, f"{self.experiment_id}_logger.pkl"), "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            loaded_logger = pickle.load(f)

        # Set the experiment_id to the max exp_id in the loaded data plus one
        loaded_logger.experiment_id = (
            max(
                loaded_logger.df_experiments["exp_id"].max(),
                loaded_logger.df_token_attr["exp_id"].max(),
                loaded_logger.df_perturbations["exp_id"].max(),
            )
            + 1
        )

        return loaded_logger

    def pretty_print(self, df: pd.DataFrame):
        # Check if code is running in Jupyter notebook
        if "IPKernelApp" in get_ipython().config:
            from IPython.display import display

            display(df)
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
                df_pivot["token", pos].astype(str)
                + "\n"
                + df_pivot["attr_score", pos].astype(str)
            )

        df_pivot = df_pivot.loc[
            :, df_pivot.columns.get_level_values(0).str.startswith("token_")
        ]
        df_pivot.reset_index(inplace=True)
        styled_df = df_pivot.style.set_properties(**{"white-space": "pre-wrap"})
        return styled_df
