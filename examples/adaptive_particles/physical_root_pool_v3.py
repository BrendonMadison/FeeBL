#!/usr/bin/env python3
"""Immutable nested-prefix capacity contract for FeeBL V3 physical roots.

This pure module allocates existing root identifiers only.  It never creates
root randomness, seeds, cards, files, runs, or representative children, and
it does not authorize a production actuator.  G4 must bind an authenticated
root generator and pool before a returned prefix can affect physics.
"""

from __future__ import annotations

from typing import Any, Sequence


PHYSICAL_ROOT_PREFIX_SCHEMA_V3 = "feebl.adaptive-convergence-v3.physical-root-prefix-extension.v1"
UINT64_MAX = (1 << 64) - 1


class PhysicalRootPoolError(RuntimeError):
    """A physical-root prefix request cannot be authorized."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhysicalRootPoolError(message)


def _root_ids(values: Any, where: str) -> tuple[int, ...]:
    _require(isinstance(values, (list, tuple)), f"{where}: expected a sequence")
    result: list[int] = []
    previous: int | None = None
    for index, value in enumerate(values):
        _require(
            isinstance(value, int) and not isinstance(value, bool),
            f"{where}[{index}]: expected an integer root ID",
        )
        _require(0 <= value <= UINT64_MAX, f"{where}[{index}]: root ID out of range")
        _require(
            previous is None or value > previous,
            f"{where}: root IDs must be strictly increasing and unique",
        )
        result.append(value)
        previous = value
    return tuple(result)


def extend_physical_root_prefix_v3(
    current_prefix: Sequence[int],
    unused_extension_pool: Sequence[int],
    requested_prefix_count: int,
    *,
    reuse_existing_roots: bool = False,
    representation_count_substitute: int | None = None,
) -> dict[str, Any]:
    """Return a stable nested prefix or fail closed before any side effect.

    Reusing an already counted root and substituting split/merge representative
    count for new physical roots are explicit errors.  A request beyond the
    authenticated unused pool reports exact exhaustion; a nonempty successful
    result remains a planning receipt, not production integration.
    """
    prefix = _root_ids(current_prefix, "$root_pool.current_prefix")
    pool = _root_ids(unused_extension_pool, "$root_pool.unused_extension_pool")
    _require(
        isinstance(requested_prefix_count, int) and not isinstance(requested_prefix_count, bool),
        "$root_pool.requested_prefix_count: expected an integer",
    )
    _require(
        len(prefix) <= requested_prefix_count <= UINT64_MAX,
        "$root_pool.requested_prefix_count: cannot shrink or overflow prefix",
    )
    _require(
        reuse_existing_roots is False,
        "physical-root reuse is forbidden",
    )
    _require(
        representation_count_substitute is None,
        "representation-count substitution cannot satisfy physical-root extension",
    )
    if prefix and pool:
        _require(
            pool[0] > prefix[-1],
            "physical-root extension pool reuses or reorders the frozen prefix",
        )
    required = requested_prefix_count - len(prefix)
    _require(
        required <= len(pool),
        "physical-root extension pool exhausted",
    )
    added = pool[:required]
    result = prefix + added
    _require(result[: len(prefix)] == prefix, "physical-root prefix stability failed")
    _require(len(result) == requested_prefix_count, "physical-root extension count failed")
    return {
        "schema": PHYSICAL_ROOT_PREFIX_SCHEMA_V3,
        "status": "planning_prefix_available_not_production_authorized",
        "prior_prefix": list(prefix),
        "added_root_ids": list(added),
        "resulting_prefix": list(result),
        "unused_pool_remaining": list(pool[required:]),
        "independent_physical_roots_added": required,
        "prefix_stable": True,
        "production_actuator_integrated": False,
        "file_seed_card_or_run_materialization_performed": False,
    }


__all__ = [
    "PHYSICAL_ROOT_PREFIX_SCHEMA_V3",
    "PhysicalRootPoolError",
    "extend_physical_root_prefix_v3",
]
