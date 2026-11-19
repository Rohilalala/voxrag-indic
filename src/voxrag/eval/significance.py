"""Is a measured difference real, or is it noise? The tests that answer that, honestly.

This module exists because of a specific mistake. A coordinate-descent parameter sweep
reported a +0.0274 nDCG@10 "win" and it was retracted, because repeating an IDENTICAL
configuration over IDENTICAL queries four times produced::

    run 1  ndcg 0.2885      run 3  ndcg 0.3213
    run 2  ndcg 0.3166      run 4  ndcg 0.2998

a spread of 0.0328 -- larger than the entire claimed improvement. The sweep had been tuning
the warm-up. Nothing in the evaluation could tell the difference, so the number shipped.

Two things fix that, and both are here.

**Compare PAIRED, not aggregate.** Comparing two mean nDCGs throws away the fact that both
configurations answered the *same* queries. Query difficulty dominates the variance of a
retrieval metric -- some queries are easy for everything, some are hopeless for everything --
and in a paired comparison that difficulty cancels exactly, because it is present in both
terms of every per-query difference. What is left is only the effect of the change. This is
why paired bootstrap and paired randomization are the standard significance tests in IR
evaluation (Smucker, Allan & Carterette, CIKM 2007) rather than an unpaired t-test: at the
n = 280--350 queries this evaluation runs, the unpaired comparison cannot resolve anything
smaller than the effects it is being asked to judge.

**Say the answer in English.** A README is read by people who do not know what a p-value is,
so `compare()` returns a verdict sentence that states outright whether the difference can be
distinguished from noise, and refuses to dress up a null result.

Both tests are two-sided, make no distributional assumption (no normality, no equal
variances -- retrieval metrics are bounded in [0, 1] and heavily zero-inflated, so normality
is plainly false), and are seeded: the same inputs and seed give bit-identical output, or the
significance test becomes another source of the noise it exists to detect.

Usage -- element i of both lists must be the SAME query, scored under each configuration::

    result = compare("dense", dense_ndcg_per_query, "hybrid", hybrid_ndcg_per_query)
    print(result.verdict)   # the sentence a report can quote verbatim
    if result.significant:  # True only when BOTH tests clear alpha
        ...

No comparison in this repository has been put through these tests yet, and this docstring
will not pretend otherwise: `voxrag.eval.retrieval.evaluate()` returns aggregates, so the
per-query score vectors these functions require are not currently persisted anywhere. The
dense-beats-hybrid gap therefore stays reported as UNRESOLVED rather than as a null result --
"not tested" and "tested, found nothing" are different claims, and only one of them is true.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

#: Run-to-run spread of nDCG@10 measured on this system with an IDENTICAL configuration over
#: IDENTICAL queries, four repeats: 0.2885 / 0.3166 / 0.3213 / 0.2998 -> spread 0.0328,
#: stdev 0.0152. The cause is warm-up (cold page cache, dense-arm timeouts on run 1), not
#: sampling, so a paired test on scores from ONE run each cannot see it: the pairing cancels
#: query difficulty, not the state of the machine. `compare()` therefore flags a difference
#: that is statistically significant but smaller than this, because such a difference is
#: exactly what the retracted +0.0274 looked like.
#:
#: Pass `noise_floor=0.0` to `compare()` when comparing a metric whose run-to-run spread has
#: NOT been measured -- do not silently reuse this number for something else.
NDCG_RUN_TO_RUN_SPREAD = 0.0328

#: Minimum queries for a paired test. Two is already too few to mean anything, but one is not
#: a comparison at all: with a single pair the randomization test has two possible outcomes
#: and the bootstrap has one, so both return an answer that is a property of the arithmetic
#: rather than of the data.
_MIN_PAIRS = 2


@dataclass(frozen=True)
class Comparison:
    """The full verdict on one A-vs-B comparison, including the sentence a human reads.

    `mean_delta` is the effect size in the metric's own units (nDCG points, say), signed as
    B minus A: positive means `name_b` scored higher. That, not a standardised effect size,
    is what a report needs -- "+0.05 nDCG@10" is directly meaningful, "d_z = 0.6" is not.
    """

    name_a: str
    name_b: str
    #: Number of queries scored under BOTH configurations.
    n: int
    mean_a: float
    mean_b: float
    #: mean(B) - mean(A). Positive => B is better.
    mean_delta: float
    bootstrap_p: float
    randomization_p: float
    #: Percentile bootstrap confidence interval for `mean_delta` at 1 - alpha.
    ci_low: float
    ci_high: float
    alpha: float
    #: True only if BOTH tests clear alpha. See `compare()` for why both.
    significant: bool
    verdict: str


def _paired_differences(scores_a: Sequence[float], scores_b: Sequence[float]) -> np.ndarray:
    """Validate two aligned per-query score lists and return B - A elementwise.

    The alignment is the whole basis of the test: element i of both arrays must be the SAME
    query. A length mismatch means the caller has silently dropped queries from one side
    (a timeout, a filtered language) and the pairing is broken, so it raises rather than
    truncating -- a truncated pairing produces a plausible number that means nothing.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"expected two flat score lists, got shapes {a.shape} and {b.shape}")
    if a.shape != b.shape:
        raise ValueError(
            f"paired test needs the same queries on both sides: got {a.size} scores for A and "
            f"{b.size} for B"
        )
    if a.size < _MIN_PAIRS:
        raise ValueError(f"need at least {_MIN_PAIRS} paired queries, got {a.size}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("scores contain NaN or inf; a failed query must be scored 0.0, not NaN")
    return b - a


def paired_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 0,
    *,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """Paired bootstrap over per-query scores -> (mean_delta, p_value, ci_low, ci_high).

    Resamples QUERIES (not scores) with replacement, carrying each query's pair of scores
    together, which is what makes the comparison paired. `mean_delta` is mean(B) - mean(A).

    The confidence interval is the percentile interval over the resampled mean differences:
    the middle 1 - alpha of the distribution of "what would this delta have been on a
    different sample of queries from the same population".

    The p-value comes from the same resamples shifted to mean zero, which is the null
    hypothesis "the two configurations are equally good" expressed as a distribution. It is
    the two-sided fraction of null resamples whose mean difference is at least as extreme as
    the observed one. Note that centring the resample means by subtracting the observed mean
    is algebraically identical to resampling the centred differences, so one resampling pass
    serves both the interval and the test.
    """
    diffs = _paired_differences(scores_a, scores_b)
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")

    n = diffs.size
    observed = float(diffs.mean())
    rng = np.random.default_rng(seed)

    # One (n_resamples, n) index matrix rather than a Python loop: ~22 MB of int64 at this
    # evaluation's 280 queries and 10k resamples, and roughly two orders of magnitude faster.
    index = rng.integers(0, n, size=(n_resamples, n))
    boot_means = diffs[index].mean(axis=1)

    ci_low, ci_high = np.percentile(
        boot_means, (100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0))
    )

    exceed = int(np.count_nonzero(np.abs(boot_means - observed) >= abs(observed)))
    # +1 to numerator and denominator: with zero exceedances the uncorrected estimate is
    # p = 0, which asserts the observed gap is IMPOSSIBLE under chance. 10k resamples cannot
    # establish that; they can establish p <= 1/(10k+1). The correction reports that bound
    # instead of a falsehood, and it is what makes identical inputs return exactly 1.0.
    p_value = (exceed + 1) / (n_resamples + 1)
    return observed, p_value, float(ci_low), float(ci_high)


def randomization_test(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_permutations: int = 10_000,
    seed: int = 0,
) -> float:
    """Fisher's two-sided permutation test on the paired differences -> p_value.

    Under the null hypothesis the two configurations are interchangeable, so for any query it
    is equally likely that the observed difference came out +d or -d. The test therefore
    enumerates that null by flipping the sign of each query's difference at random, and asks
    how often pure sign noise produces a mean difference at least as large in magnitude as
    the one actually measured.

    This assumes nothing at all about the distribution of the scores -- no normality, no
    symmetry, no equal variance -- which matters because retrieval metrics are bounded and
    zero-inflated (a query where nothing relevant was retrieved scores exactly 0 under every
    configuration, contributing an exact 0 difference that correctly counts as no evidence).

    2**n sign patterns is astronomically many at n = 280, so this samples `n_permutations` of
    them uniformly: a Monte Carlo permutation test, with the same +1 correction as the
    bootstrap so the reported p is never a false 0.

    It is reported ALONGSIDE the bootstrap rather than instead of it because the two make
    different assumptions; agreement is evidence, disagreement is a warning.
    """
    diffs = _paired_differences(scores_a, scores_b)
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")

    observed = abs(float(diffs.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_permutations, diffs.size)) * 2 - 1
    means = (signs * diffs).mean(axis=1)
    exceed = int(np.count_nonzero(np.abs(means) >= observed))
    return (exceed + 1) / (n_permutations + 1)


def _chance_phrase(p: float) -> str:
    """A p-value as odds someone who has never seen one can read.

    Deliberately rounds DOWN in strength, in both directions. Below 1e-4 it says "1 time in
    10,000" rather than extrapolating past what 10k resamples can resolve; above 1e-2 it
    rounds the count of noise-produced gaps UP (ceil, not round) so the chance of the result
    being noise is never understated. Ordinary rounding would report p = 0.045 as "about 4
    times in 100", which reads as a stronger result than was measured -- and flattering the
    number by a hair is the exact failure this module exists to prevent.
    """
    for threshold, phrase in (
        (1e-4, "fewer than 1 time in 10,000"),
        (1e-3, "fewer than 1 time in 1,000"),
        (1e-2, "fewer than 1 time in 100"),
    ):
        if p < threshold:
            return phrase
    # round(..., 6) before the ceil is not cosmetic: 0.30 * 100 is 30.000000000000004 in
    # binary floating point, and a bare ceil would turn a p of exactly 0.30 into "31 times".
    times = math.ceil(round(p * 100, 6))
    return f"about {times} time{'' if times == 1 else 's'} in 100"


def _verdict(
    *,
    name_a: str,
    name_b: str,
    n: int,
    mean_a: float,
    mean_b: float,
    delta: float,
    bootstrap_p: float,
    randomization_p: float,
    ci_low: float,
    ci_high: float,
    alpha: float,
    significant: bool,
    noise_floor: float,
) -> str:
    """The plain-English answer. No jargon that is not immediately explained in the sentence."""
    if delta == 0.0:
        return (
            f"{name_a} and {name_b} scored exactly the same on average ({mean_a:.4f} over "
            f"{n:,} queries). There is no difference to report."
        )

    better, worse = (name_b, name_a) if delta > 0.0 else (name_a, name_b)
    head = (
        f"{better} scored {abs(delta):.4f} higher than {worse} on the same {n:,} queries "
        f"({name_a} {mean_a:.4f}, {name_b} {mean_b:.4f})."
    )
    # Quote the weaker of the two tests: reporting whichever came out more flattering is
    # p-hacking with extra steps.
    chance = _chance_phrase(max(bootstrap_p, randomization_p))
    interval = (
        f"The plausible range for the true difference runs from {ci_low:+.4f} to {ci_high:+.4f}"
    )

    if not significant:
        parts = [
            head,
            f"That gap is INDISTINGUISHABLE FROM NOISE: if the two were really equally good, a "
            f"gap at least this big would still turn up {chance}, purely from which queries "
            f"happened to be in the sample.",
            f"{interval}"
            + (
                ", which includes zero -- a re-run could plausibly reverse the winner."
                if ci_low <= 0.0 <= ci_high
                else "."
            ),
        ]
        if min(bootstrap_p, randomization_p) < alpha:
            parts.append(
                f"The two tests also disagree (bootstrap p={bootstrap_p:.4f}, randomization "
                f"p={randomization_p:.4f}), which is itself a reason not to claim a win."
            )
        parts.append("Do not report this as an improvement.")
        return " ".join(parts)

    parts = [
        head,
        f"That gap is REAL: if the two were really equally good, a gap this big would turn up "
        f"{chance} (bootstrap p={bootstrap_p:.4f}, randomization p={randomization_p:.4f}).",
        f"{interval}, and it stays on one side of zero.",
    ]
    if 0.0 < abs(delta) < noise_floor:
        parts.append(
            f"CAVEAT: {abs(delta):.4f} is still smaller than the {noise_floor:.4f} spread "
            f"measured between four repeats of an IDENTICAL configuration on IDENTICAL "
            f"queries. These tests can only say that these two sets of scores differ; they "
            f"cannot say the configuration caused it, because run order and a cold cache move "
            f"the number by more than this on their own. Re-run both configurations several "
            f"times before reporting it."
        )
    else:
        parts.append("This one is safe to report.")
    return " ".join(parts)


def compare(
    name_a: str,
    scores_a: Sequence[float],
    name_b: str,
    scores_b: Sequence[float],
    alpha: float = 0.05,
    *,
    n_resamples: int = 10_000,
    n_permutations: int = 10_000,
    seed: int = 0,
    noise_floor: float = NDCG_RUN_TO_RUN_SPREAD,
) -> Comparison:
    """Run both paired tests on one metric and return a result a report can quote verbatim.

    `scores_a` and `scores_b` are per-query scores of the same metric, aligned query by query:
    element i of each must be the same query. Order the arguments however you like; the sign
    of `mean_delta` says which name won.

    A difference counts as significant only when BOTH tests clear `alpha`. They rarely
    disagree, and when they do it is on borderline effects -- exactly the cases where the
    honest answer is "not established". Taking the better of two p-values would double the
    false-positive rate for free, which is how the retracted +0.0274 happened.

    `noise_floor` is the run-to-run spread of this metric measured on identical inputs. A
    significant difference smaller than it is flagged in the verdict, because the pairing
    cancels query difficulty but cannot cancel a cold cache. Pass 0.0 to disable the check for
    a metric whose run-to-run spread has not been measured.
    """
    delta, bootstrap_p, ci_low, ci_high = paired_bootstrap(
        scores_a, scores_b, n_resamples, seed, alpha=alpha
    )
    randomization_p = randomization_test(scores_a, scores_b, n_permutations, seed)

    n = len(scores_a)
    mean_a = float(np.mean(np.asarray(scores_a, dtype=np.float64)))
    mean_b = float(np.mean(np.asarray(scores_b, dtype=np.float64)))
    significant = bootstrap_p < alpha and randomization_p < alpha

    return Comparison(
        name_a=name_a,
        name_b=name_b,
        n=n,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_delta=delta,
        bootstrap_p=bootstrap_p,
        randomization_p=randomization_p,
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=alpha,
        significant=significant,
        verdict=_verdict(
            name_a=name_a,
            name_b=name_b,
            n=n,
            mean_a=mean_a,
            mean_b=mean_b,
            delta=delta,
            bootstrap_p=bootstrap_p,
            randomization_p=randomization_p,
            ci_low=ci_low,
            ci_high=ci_high,
            alpha=alpha,
            significant=significant,
            noise_floor=noise_floor,
        ),
    )
