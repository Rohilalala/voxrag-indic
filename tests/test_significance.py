"""Significance testing: does the evaluation know the difference between a win and noise?

These tests are the reason the module exists, so they are written as claims about behaviour
that a fabricated result would have to violate:

* identical inputs must come back p = 1.0 -- not "p = 0.03 by luck of the seed";
* a large shift must be detected;
* a shift smaller than this system's measured run-to-run spread must NOT be, at the query
  counts this evaluation actually runs (n = 280);
* the confidence interval must contain the true delta about 95% of the time on data whose
  true delta is known by construction;
* and the whole thing must be bit-reproducible, or the test for noise is itself noisy.

Everything runs on synthetic scores from a seeded numpy Generator: hermetic, no corpus, no
network, no index, and fast enough to keep the 10k-resample default in the paths where the
default is what matters.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxrag.eval.significance import (
    NDCG_RUN_TO_RUN_SPREAD,
    Comparison,
    _chance_phrase,
    compare,
    paired_bootstrap,
    randomization_test,
)

#: The evaluation's real query count (350 graded queries, 280 after per-language balancing).
#: Effect sizes below are chosen against this n on purpose -- "detectable" is a statement
#: about a sample size, and asserting it at n = 10,000 would prove nothing about this system.
N_QUERIES = 280

#: Per-query stdev of the DIFFERENCE between two retrieval configurations. Two configurations
#: agree on most queries and disagree sharply on a few, which is what a 0.15 spread over
#: nDCG@10 values in [0, 1] represents. At n = 280 this puts the standard error of the mean
#: difference at 0.15/sqrt(280) = 0.0090.
DIFF_SPREAD = 0.15


def _synthetic_pair(
    delta: float, *, n: int = N_QUERIES, seed: int = 0, spread: float = DIFF_SPREAD
) -> tuple[list[float], list[float]]:
    """Two per-query score lists whose TRUE mean difference is exactly `delta`.

    The shared `base` is the query-difficulty term that pairing is supposed to cancel: it has
    a far larger spread (Beta(2,3) over [0,1], stdev ~0.2) than the effect being detected, so
    an unpaired comparison of these two lists would drown. Scores are deliberately not
    clipped back into [0, 1] -- clipping would bias the mean difference away from `delta` and
    quietly break the interval-coverage test below.
    """
    rng = np.random.default_rng(seed)
    base = rng.beta(2.0, 3.0, size=n)
    b = base + delta + rng.normal(0.0, spread, size=n)
    return base.tolist(), b.tolist()


def test_identical_scores_give_zero_effect_and_p_one() -> None:
    """The single most important property: no difference must read as no difference."""
    scores, _ = _synthetic_pair(0.0, seed=1)

    delta, p_boot, ci_low, ci_high = paired_bootstrap(scores, scores)
    assert delta == 0.0
    assert p_boot == 1.0
    assert ci_low == 0.0 and ci_high == 0.0

    assert randomization_test(scores, scores) == 1.0

    result = compare("same", scores, "same-again", scores)
    assert result.significant is False
    assert result.mean_delta == 0.0
    assert result.bootstrap_p == 1.0
    assert result.randomization_p == 1.0
    assert "no difference to report" in result.verdict


def test_large_shift_is_detected() -> None:
    """+0.08 nDCG at n=280 is ~9 standard errors. If this is missed, the test is broken."""
    a, b = _synthetic_pair(0.08, seed=2)

    delta, p_boot, ci_low, _ci_high = paired_bootstrap(a, b)
    assert delta == pytest.approx(0.08, abs=0.02)
    assert p_boot < 0.001
    assert ci_low > 0.0

    assert randomization_test(a, b) < 0.001

    result = compare("baseline", a, "tuned", b)
    assert result.significant is True
    assert "REAL" in result.verdict
    assert "safe to report" in result.verdict


def test_shift_below_the_noise_floor_is_not_significant() -> None:
    """+0.002 is a fifth of a standard error at n=280. It must not read as a win.

    This is the shape of the retracted +0.0274: a small positive number that looks like
    progress on a leaderboard and is indistinguishable from resampling the query set.
    """
    a, b = _synthetic_pair(0.002, seed=3)

    _delta, p_boot, ci_low, ci_high = paired_bootstrap(a, b)
    assert p_boot > 0.05
    assert ci_low < 0.0 < ci_high

    assert randomization_test(a, b) > 0.05

    result = compare("baseline", a, "tuned", b)
    assert result.significant is False
    assert "INDISTINGUISHABLE FROM NOISE" in result.verdict
    assert "Do not report this as an improvement." in result.verdict


def test_significant_but_sub_noise_floor_difference_is_flagged() -> None:
    """Statistically significant is not the same as trustworthy, and the verdict says so.

    With a small per-query spread (0.01) even a +0.02 shift clears both tests comfortably --
    but 0.02 is below the 0.0328 run-to-run spread measured on identical configurations, so
    reporting it as a configuration win would repeat the original mistake.
    """
    a, b = _synthetic_pair(0.02, seed=4, spread=0.01)
    result = compare("run-a", a, "run-b", b)

    assert result.significant is True
    assert abs(result.mean_delta) < NDCG_RUN_TO_RUN_SPREAD
    assert "CAVEAT" in result.verdict
    assert f"{NDCG_RUN_TO_RUN_SPREAD:.4f}" in result.verdict

    # The flag is a property of the measured floor, not a hardcoded string: disabling it
    # leaves an otherwise identical verdict without the caveat.
    unflagged = compare("run-a", a, "run-b", b, noise_floor=0.0)
    assert unflagged.significant is True
    assert "CAVEAT" not in unflagged.verdict


def test_confidence_interval_brackets_the_true_delta() -> None:
    """A 95% interval must contain the true delta about 95% of the time, and it does.

    Coverage is checked over 40 independent synthetic datasets whose true delta is known by
    construction. This implementation covers it 39 times out of 40; the assertion is set at
    34 because the percentile bootstrap is only asymptotically exact and the binomial spread
    on 40 trials at p=0.95 is itself +-1.4, so a tighter bound would fail on an unlucky but
    correct implementation. It still fails loudly on an interval that is systematically too
    narrow or mis-centred.
    """
    true_delta = 0.03
    covered = 0
    for data_seed in range(40):
        a, b = _synthetic_pair(true_delta, seed=100 + data_seed)
        _delta, _p, ci_low, ci_high = paired_bootstrap(a, b, n_resamples=2_000, seed=data_seed)
        if ci_low <= true_delta <= ci_high:
            covered += 1
    assert covered >= 34, f"interval covered the true delta only {covered}/40 times"


def test_interval_is_centred_on_the_observed_delta() -> None:
    a, b = _synthetic_pair(0.05, seed=5)
    delta, _p, ci_low, ci_high = paired_bootstrap(a, b)
    assert ci_low < delta < ci_high
    # Roughly symmetric: the bootstrap distribution of a mean is near-normal even when the
    # underlying scores are not, so a badly lopsided interval means the resampling is wrong.
    assert (delta - ci_low) == pytest.approx(ci_high - delta, rel=0.3)


def test_both_functions_are_deterministic_under_a_fixed_seed() -> None:
    """Two runs of the same comparison must agree EXACTLY, not approximately."""
    a, b = _synthetic_pair(0.04, seed=6)

    assert paired_bootstrap(a, b, 2_000, 7) == paired_bootstrap(a, b, 2_000, 7)
    assert randomization_test(a, b, 2_000, 7) == randomization_test(a, b, 2_000, 7)

    first = compare("a", a, "b", b, n_resamples=2_000, n_permutations=2_000, seed=11)
    second = compare("a", a, "b", b, n_resamples=2_000, n_permutations=2_000, seed=11)
    assert first == second
    assert isinstance(first, Comparison)


def test_seed_actually_changes_the_resampling() -> None:
    """Guards against a "deterministic" implementation that ignores the seed entirely."""
    a, b = _synthetic_pair(0.04, seed=6)
    _d1, _p1, low_1, high_1 = paired_bootstrap(a, b, 2_000, 0)
    _d2, _p2, low_2, high_2 = paired_bootstrap(a, b, 2_000, 1)
    assert (low_1, high_1) != (low_2, high_2)


def test_swapping_the_arguments_flips_the_sign_and_keeps_the_verdict() -> None:
    """A vs B and B vs A are the same experiment; only the direction may change."""
    a, b = _synthetic_pair(0.08, seed=8)

    forward = compare("baseline", a, "tuned", b)
    reverse = compare("tuned", b, "baseline", a)

    assert reverse.mean_delta == pytest.approx(-forward.mean_delta)
    assert reverse.ci_low == pytest.approx(-forward.ci_high)
    assert reverse.ci_high == pytest.approx(-forward.ci_low)
    assert reverse.significant == forward.significant
    # Both name the same winner despite the swapped argument order.
    assert forward.verdict.startswith("tuned scored")
    assert reverse.verdict.startswith("tuned scored")


def test_p_values_are_bounded_and_never_exactly_zero() -> None:
    """10k resamples cannot prove impossibility, so p must never come back as 0."""
    a, b = _synthetic_pair(2.0, seed=9)  # absurdly large effect
    _delta, p_boot, _low, _high = paired_bootstrap(a, b, 1_000, 0)
    p_rand = randomization_test(a, b, 1_000, 0)

    for p in (p_boot, p_rand):
        assert 0.0 < p <= 1.0
        assert p == pytest.approx(1.0 / 1_001)


def test_pairing_resolves_an_effect_an_unpaired_comparison_cannot() -> None:
    """The justification for the whole module, asserted rather than claimed.

    The shared query-difficulty term (Beta(2,3), stdev ~0.2) is five times the +0.04 effect.
    Pairing cancels it and the effect comes out clearly significant (p = 0.0010 on this
    dataset); shuffling one side -- same values, same two means, same marginal distributions,
    only the query alignment destroyed -- buries the identical effect in that difficulty
    variance and it is no longer detectable (p = 0.11).
    """
    a, b = _synthetic_pair(0.04, seed=11)
    assert compare("a", a, "b", b).significant is True

    shuffled = list(b)
    np.random.default_rng(0).shuffle(shuffled)
    assert compare("a", a, "b-unpaired", shuffled).significant is False


def test_mismatched_or_degenerate_inputs_raise() -> None:
    a, b = _synthetic_pair(0.05, seed=12)

    with pytest.raises(ValueError, match="same queries on both sides"):
        paired_bootstrap(a, b[:-1])
    with pytest.raises(ValueError, match="at least 2 paired queries"):
        paired_bootstrap([0.5], [0.6])
    with pytest.raises(ValueError, match="NaN or inf"):
        randomization_test([0.5, float("nan"), 0.2], [0.5, 0.4, 0.2])
    with pytest.raises(ValueError, match="n_resamples"):
        paired_bootstrap(a, b, 0)
    with pytest.raises(ValueError, match="n_permutations"):
        randomization_test(a, b, 0)
    with pytest.raises(ValueError, match="alpha"):
        paired_bootstrap(a, b, alpha=1.0)


def test_chance_phrase_never_understates_the_chance_of_noise() -> None:
    """The one sentence a non-statistician actually reads. Grammar and direction both matter.

    Rounding is UP, so a p of 0.045 reads as "5 times in 100" rather than the flattering "4".
    """
    assert _chance_phrase(0.045) == "about 5 times in 100"
    assert _chance_phrase(0.30) == "about 30 times in 100"  # not 31: 0.30 * 100 > 30 in float
    assert _chance_phrase(1.0) == "about 100 times in 100"
    # p exactly 0.01 is the only value that reaches the singular, and it is reachable:
    # (exceed + 1) / (n_resamples + 1) = 10/1000 with n_resamples=999.
    assert _chance_phrase(0.01) == "about 1 time in 100"
    # Below 1% the odds are stated as a bound, never extrapolated past the resample count.
    assert _chance_phrase(0.004) == "fewer than 1 time in 100"
    assert _chance_phrase(0.0004) == "fewer than 1 time in 1,000"
    assert _chance_phrase(1 / 10_001) == "fewer than 1 time in 10,000"
    assert _chance_phrase(0.0) == "fewer than 1 time in 10,000"


def test_verdict_is_readable_without_knowing_what_a_p_value_is() -> None:
    """The verdict is what lands in a README, so it must stand alone."""
    a, b = _synthetic_pair(0.001, seed=13)
    verdict = compare("dense", a, "hybrid", b).verdict

    assert "dense" in verdict and "hybrid" in verdict
    # States the answer in words, gives the odds in words, and quantifies the uncertainty.
    assert "INDISTINGUISHABLE FROM NOISE" in verdict
    assert "times in 100" in verdict
    assert "plausible range for the true difference" in verdict
    assert verdict.endswith("Do not report this as an improvement.")
