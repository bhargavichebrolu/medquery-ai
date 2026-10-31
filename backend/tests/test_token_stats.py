import pytest

from medquery.embeddings import summarise_token_lengths


def test_statistics_and_truncation() -> None:
    ids = ["a", "b", "c", "d", "e"]
    stats = summarise_token_lengths(ids, [10, 300, 256, 257, 100], max_seq_length=256)
    assert (stats.count, stats.min, stats.max, stats.median) == (5, 10, 300, 256.0)
    assert stats.mean == 184.6
    assert stats.truncated_count == 2  # exactly 256 fits
    assert stats.truncated_fraction == 0.4
    assert stats.tokens_lost_total == 44 + 1
    assert [(w.item_id, w.tokens, w.tokens_lost) for w in stats.worst] == [
        ("b", 300, 44),
        ("d", 257, 1),
    ]


def test_worst_list_is_capped() -> None:
    stats = summarise_token_lengths([str(i) for i in range(20)], [500] * 20, 100, top_n=3)
    assert len(stats.worst) == 3 and stats.truncated_count == 20


def test_unknown_limit_means_no_truncation() -> None:
    stats = summarise_token_lengths(["a"], [10_000], max_seq_length=None)
    assert stats.truncated_count == 0 and stats.worst == ()


def test_invalid_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        summarise_token_lengths(["a"], [1, 2], 10)
    with pytest.raises(ValueError):
        summarise_token_lengths([], [], 10)
