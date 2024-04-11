from typing import List

import torch
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text


class RichTablePrinter:
    def __init__(self, max_abs_attr_val):
        self.max_abs_attr_val = max_abs_attr_val
        self.console = Console()

    def get_color(self, value):
        """Get color intensity based on the value compared to the max_value."""
        intensity = abs(value) / self.max_abs_attr_val
        if intensity < 0.25:
            return "green"
        elif intensity < 0.5:
            return "yellow"
        else:
            return "red"

    def print_attribution_table(
        self,
        token_list: List[str],
        attr_scores: torch.Tensor,
        token_ids: torch.Tensor,
        generation_length: int,
    ):
        input_length = token_ids.shape[0] - generation_length
        table = Table(show_header=True, header_style="bold", box=box.SIMPLE)

        # Construct the header text with color coding
        header_text = Text()
        for i, token in enumerate(token_list):
            if i < input_length:
                header_text.append(token)
            else:
                header_text.append(token, style="blue")
        self.console.print("\n")
        self.console.print(header_text)

        # Transpose the attr_scores for table representation
        attr_scores = attr_scores.T

        # Add columns (axis labels for y-axis)
        table.add_column("")
        for i in range(input_length, len(token_list)):
            table.add_column(token_list[i], justify="right")

        # Add rows with data (axis labels for x-axis will be the first column)
        for j in range(attr_scores.shape[0]):
            row = [token_list[j]]  # First column with the x-axis label
            for i in range(attr_scores.shape[1]):
                value = attr_scores[j, i].item()
                color = self.get_color(value)
                row.append(f"[{color}]{value:.4f}[/]")
            table.add_row(*row)

        # Print the table to the console
        self.console.print(table)
