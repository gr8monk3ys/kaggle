#!/usr/bin/env python3
"""Local competition lab: reproducible model runs, one module per competition.

The interface is unchanged from the single-file version::

    BENCHMARKS[slug](data_dir, folds, write_submission) -> LabResult

What changed is where the implementations live. Eight competitions shared one
4,500-line namespace, distinguished only by a private-helper prefix convention
(``_playground_``, ``_march_``, …) that was doing a module boundary's job — so
touching store-sales meant navigating march-mania, and the tests reached past the
interface to monkeypatch seven private helpers per benchmark.

The registry holds dotted paths rather than imported callables: resolving a
benchmark imports only its module, so listing the benchmarks does not drag in
pandas, numpy and fourteen sklearn submodules eight times over.
"""

from __future__ import annotations


import argparse
import importlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps pandas off the import path
    from kaggle_portfolio.notebooks.competition_lab.runner import LabResult

from kaggle_portfolio.shared.deps import Deps
from kaggle_portfolio.shared.errors import CommandError


def __getattr__(name: str):
    """Re-export the harness lazily.

    ``runner`` imports pandas, so importing it here would make merely listing the
    benchmarks pay for it — the cost this package was split up to avoid.
    """
    if name in {
        "LabResult",
        "_ensure_data",
        "_print_benchmarks",
        "_save_summary",
        "_submit",
    }:
        from kaggle_portfolio.notebooks.competition_lab import runner

        return getattr(runner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["BENCHMARKS", "LabResult", "load_benchmark", "main", "parse_args"]

#: slug -> "module:function". Not read directly — see the BENCHMARKS mapping below.
_TARGETS: dict[str, str] = {
    "titanic": "titanic:benchmark_titanic",
    "spaceship-titanic": "spaceship:benchmark_spaceship",
    "nlp-getting-started": "nlp:benchmark_nlp",
    "playground-series-s6e3": "playground_telco:benchmark_playground_telco",
    "house-prices-advanced-regression-techniques": "house_prices:benchmark_house_prices",
    "store-sales-time-series-forecasting": "store_sales:benchmark_store_sales",
    "deep-past-initiative-machine-translation": "deep_past:benchmark_deep_past",
    "march-machine-learning-mania-2026": "march_mania:benchmark_march_mania",
}


def load_benchmark(slug: str) -> Callable[..., "LabResult"]:
    """Import and return one competition's benchmark function."""
    try:
        target = _TARGETS[slug]
    except KeyError:
        known = ", ".join(sorted(_TARGETS))
        raise KeyError(f"Unknown competition {slug!r}. Known: {known}") from None
    module_name, _, func = target.partition(":")
    module = importlib.import_module(
        f"kaggle_portfolio.notebooks.competition_lab.{module_name}"
    )
    return getattr(module, func)


class _LazyBenchmarks(Mapping):
    """``BENCHMARKS[slug]`` returns the benchmark function, importing it on access.

    The split briefly turned this into a dict of dotted-path strings, which
    silently broke the documented contract — ``BENCHMARKS[slug](data_dir, folds,
    write)`` raised TypeError because the value was a str. A Mapping keeps the
    contract and the laziness: nothing is imported until a slug is looked up.
    """

    def __getitem__(self, slug: str) -> Callable[..., "LabResult"]:
        return load_benchmark(slug)

    def __iter__(self):
        return iter(_TARGETS)

    def __len__(self) -> int:
        return len(_TARGETS)

    def __contains__(self, slug: object) -> bool:
        return slug in _TARGETS

    def __repr__(self) -> str:
        return f"<BENCHMARKS {sorted(_TARGETS)}>"


#: slug -> benchmark callable. Iterating or ``len()`` imports nothing.
BENCHMARKS = _LazyBenchmarks()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local competition benchmarks and generate Kaggle submissions."
    )
    parser.add_argument("slug", choices=sorted(BENCHMARKS))
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--write-submission",
        action="store_true",
        help="Write the best local submission CSV",
    )
    parser.add_argument(
        "--submit", action="store_true", help="Submit the generated CSV to Kaggle"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download competition data even if cached locally",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    from kaggle_portfolio.notebooks.competition_lab.runner import (
        _ensure_data,
        _print_benchmarks,
        _save_summary,
        _submit,
    )

    args = parse_args(argv)
    # Effects are ON: fetching competition data is the lab's job, and gating it
    # behind --submit meant a cold cache silently produced an empty data dir.
    # Submission is separately gated by --submit below, which is the user's
    # explicit opt-in.
    deps = deps or Deps.resolve()
    data_dir = _ensure_data(deps, args.slug, force_download=args.force_download)
    bench_fn = load_benchmark(args.slug)
    result = bench_fn(
        data_dir, args.cv_folds, write_submission=(args.write_submission or args.submit)
    )
    _save_summary(result)
    _print_benchmarks(result)

    if args.submit:
        if result.submission_path is None:
            raise CommandError("No submission file was generated.")
        message = f"Local {result.best_model} baseline via competition-lab ({result.metric_name}={result.best_score:.5f})"
        _submit(deps, args.slug, result.submission_path, message)

    return 0
