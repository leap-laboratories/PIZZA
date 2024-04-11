import torch

from attribution.visualization import RichTablePrinter


def test_get_color():
    printer = RichTablePrinter(max_abs_attr_val=1.0)

    assert printer.get_color(0.1) == "green"
    assert printer.get_color(0.3) == "yellow"
    assert printer.get_color(0.7) == "red"


def test_print_attribution_table(capsys):
    printer = RichTablePrinter(max_abs_attr_val=1.0)
    token_list = [
        "the",
        " quick",
        " brown",
        " fox",
        " jumps",
        " over",
        " the",
        " lazy",
        " dog",
    ]
    attr_scores = torch.tensor(
        [
            [0.1, 0.6, 0.1, 0.6, 0.1, 0.6, 0.1, 0.6, 0.1],
            [0.2, 0.7, 0.2, 0.7, 0.2, 0.7, 0.2, 0.7, 0.2],
            [0.3, 0.8, 0.3, 0.8, 0.3, 0.8, 0.3, 0.8, 0.3],
            [0.4, 0.9, 0.4, 0.9, 0.4, 0.9, 0.4, 0.9, 0.4],
            [0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5],
        ]
    )  # generation_length x token_list.length
    token_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    generation_length = 5

    printer.print_attribution_table(
        token_list, attr_scores, token_ids, generation_length
    )

    captured = capsys.readouterr()
    assert "the quick brown fox jumps over the lazy dog" in captured.out

    assert "jumps     over      the     lazy      dog" in captured.out
    assert "─────────────────────────────────────────────────────" in captured.out
    assert "the      0.1000   0.2000   0.3000   0.4000   0.5000" in captured.out
    assert "quick   0.6000   0.7000   0.8000   0.9000   1.0000" in captured.out
    assert "brown   0.1000   0.2000   0.3000   0.4000   0.5000" in captured.out
    assert "fox     0.6000   0.7000   0.8000   0.9000   1.0000" in captured.out
    assert "jumps   0.1000   0.2000   0.3000   0.4000   0.5000" in captured.out
    assert "over    0.6000   0.7000   0.8000   0.9000   1.0000" in captured.out
    assert "the     0.1000   0.2000   0.3000   0.4000   0.5000" in captured.out
    assert "lazy    0.6000   0.7000   0.8000   0.9000   1.0000" in captured.out
    assert "dog     0.1000   0.2000   0.3000   0.4000   0.5000" in captured.out
