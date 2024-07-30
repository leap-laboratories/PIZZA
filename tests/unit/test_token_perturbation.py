from unittest.mock import Mock

import numpy as np
import pytest

from attribution.token_perturbation import (
    calculate_chunk_size,
    get_masks,
    get_units_from_prompt,
    split_mask,
)


@pytest.fixture
def mock_tokenizer():
    tokenizer = Mock()
    tokenizer.tokenize.side_effect = lambda x: [x]
    tokenizer.encode.side_effect = lambda x, add_special_tokens: list(range(len(x)))
    return tokenizer


# Note we only test "word" unit definition here, since "token" would require a real tokenizer
@pytest.mark.parametrize(
    "input_text, unit_definition, expected_tokens, expected_ids",
    [
        ("hello world", "word", [["hello"], [" world"]], [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]]),
        (
            "test input text",
            "word",
            [["test"], [" input"], [" text"]],
            [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4]],
        ),
    ],
)
def test_get_units_from_prompt(
    mock_tokenizer, input_text, unit_definition, expected_tokens, expected_ids
):
    result_tokens, result_ids = get_units_from_prompt(input_text, mock_tokenizer, unit_definition)
    assert result_tokens == expected_tokens
    assert result_ids == expected_ids


def test_get_masks():
    input_size = 10
    window_size = 4
    stride = 2

    expected_masks = [
        np.array([True, True, False, False, False, False, False, False, False, False]),
        np.array([True, True, True, True, False, False, False, False, False, False]),
        np.array([False, False, True, True, True, True, False, False, False, False]),
        np.array([False, False, False, False, True, True, True, True, False, False]),
        np.array([False, False, False, False, False, False, True, True, True, True]),
        np.array([False, False, False, False, False, False, False, False, True, True]),
    ]

    masks = get_masks(input_size, window_size, stride)

    assert len(masks) == len(expected_masks)
    for mask, expected_mask in zip(masks, expected_masks):
        assert np.array_equal(mask, expected_mask)


def test_get_masks_with_none_stride():
    input_size = 10
    window_size = 4
    expected_masks = [
        np.array([True, True, False, False, False, False, False, False, False, False]),
        np.array([False, False, True, True, True, True, False, False, False, False]),
        np.array([False, False, False, False, False, False, True, True, True, True]),
    ]

    masks = get_masks(input_size, window_size, stride=None)

    assert len(masks) == len(expected_masks)
    for mask, expected_mask in zip(masks, expected_masks):
        assert np.array_equal(mask, expected_mask)


test_split_mask_cases = [
    (
        np.array([False, True, True, True, True, True, False, False, False, False]),
        np.array([False, True, True, False, False, False, False, False, False, False]),
        np.array([False, False, False, True, True, True, False, False, False, False]),
    ),
    (
        np.array([False, False, False, False, False, False, False, False, True, True, True]),
        np.array([False, False, False, False, False, False, False, False, True, False, False]),
        np.array([False, False, False, False, False, False, False, False, False, True, True]),
    ),
    (
        np.array([True, True, True, True, True]),
        np.array([True, True, False, False, False]),
        np.array([False, False, True, True, True]),
    ),
    (
        np.array([False, False, False, False, False]),
        np.array([False, False, False, False, False]),
        np.array([False, False, False, False, False]),
    ),
]


@pytest.mark.parametrize("mask, expected_mask1, expected_mask2", test_split_mask_cases)
def test_split_mask(mask, expected_mask1, expected_mask2):
    mask1, mask2 = split_mask(mask)

    assert np.array_equal(mask1, expected_mask1)
    assert np.array_equal(mask2, expected_mask2)


def test_calculate_chunk_size_with_fraction():
    token_count = 100
    fraction = 0.5
    expected_chunk_size = 50

    chunk_size = calculate_chunk_size(token_count, fraction=fraction)

    assert chunk_size == expected_chunk_size


def test_calculate_chunk_size_with_num_chunks():
    token_count = 100
    num_chunks = 4
    expected_chunk_size = 25

    chunk_size = calculate_chunk_size(token_count, num_chunks=num_chunks)

    assert chunk_size == expected_chunk_size


def test_calculate_chunk_size_with_min_size():
    token_count = 100
    fraction = 0.1
    min_size = 20
    expected_chunk_size = 20

    chunk_size = calculate_chunk_size(token_count, fraction=fraction, min_size=min_size)

    assert chunk_size == expected_chunk_size


def test_calculate_chunk_size_with_max_size():
    token_count = 100
    fraction = 0.9
    max_size = 80
    expected_chunk_size = 80

    chunk_size = calculate_chunk_size(token_count, fraction=fraction, max_size=max_size)

    assert chunk_size == expected_chunk_size


def test_calculate_chunk_size_raises_error():
    token_count = 100
    with pytest.raises(ValueError):
        calculate_chunk_size(token_count)
