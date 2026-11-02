#!/usr/bin/env python3
"""Standalone adaptive-macroparticle algorithm comparator.

This module is intentionally independent of FeeBL production C++.  It compares
four representation algorithms on deterministic synthetic weighted phase
space and emits auditable JSON/CSV evidence.  It must not be imported by the
simulation or described as a production implementation.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA = "feebl.adaptive_particles.prototype_comparison.v1"
MASS = 1.0


class AlgorithmRollback(RuntimeError):
    """A proposal failed a hard precondition and leaves its input unchanged."""


@dataclass(frozen=True)
class Particle:
    uid: int
    x: float
    y: float
    px: float
    py: float
    pz: float
    energy: float
    weight: float
    cohort: int = 0
    ancestry: Tuple[Tuple[int, float], ...] = ()
    ancestry_variance_mass: float = 0.0


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def validate(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise AlgorithmRollback("CIC grid needs at least two cells per axis")
        values = (self.xmin, self.xmax, self.ymin, self.ymax)
        if not all(math.isfinite(value) for value in values):
            raise AlgorithmRollback("CIC grid bounds must be finite")
        if not (self.xmax > self.xmin and self.ymax > self.ymin):
            raise AlgorithmRollback("CIC grid bounds must be ordered")

    @property
    def dx(self) -> float:
        return (self.xmax - self.xmin) / self.nx

    @property
    def dy(self) -> float:
        return (self.ymax - self.ymin) / self.ny

    def cic(self, x: float, y: float) -> Tuple[int, int, float, float]:
        self.validate()
        ux = (x - self.xmin) / self.dx - 0.5
        uy = (y - self.ymin) / self.dy - 0.5
        ix = math.floor(ux)
        iy = math.floor(uy)
        fx = ux - ix
        fy = uy - iy
        if ix < 0 or iy < 0 or ix + 1 >= self.nx or iy + 1 >= self.ny:
            raise AlgorithmRollback(f"particle ({x:.6g},{y:.6g}) escapes the complete CIC stencil")
        return int(ix), int(iy), fx, fy

    def centre(self, ix: int, iy: int) -> Tuple[float, float]:
        return (
            self.xmin + (ix + 0.5) * self.dx,
            self.ymin + (iy + 0.5) * self.dy,
        )


@dataclass
class Representation:
    algorithm: str
    particles: List[Particle]
    exact_ancestry: bool
    committed: bool = True
    reason: str = "accepted"
    details: Dict[str, object] = field(default_factory=dict)


def physical_energy(px: float, py: float, pz: float, mass: float = MASS) -> float:
    return math.sqrt(mass * mass + px * px + py * py + pz * pz)


def validate_particles(particles: Sequence[Particle]) -> None:
    if not particles:
        raise AlgorithmRollback("particle ensemble is empty")
    seen = set()
    for particle in particles:
        scalars = (
            particle.x,
            particle.y,
            particle.px,
            particle.py,
            particle.pz,
            particle.energy,
            particle.weight,
            particle.ancestry_variance_mass,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise AlgorithmRollback("particle ensemble contains a non-finite value")
        if particle.weight <= 0.0:
            raise AlgorithmRollback("particle weights must be strictly positive")
        if particle.uid in seen:
            raise AlgorithmRollback("input root IDs must be unique")
        seen.add(particle.uid)
        shell = physical_energy(particle.px, particle.py, particle.pz)
        if particle.energy + 1.0e-12 * max(1.0, shell) < shell:
            raise AlgorithmRollback("particle energy lies below its mass shell")


def as_roots(particles: Iterable[Particle]) -> List[Particle]:
    roots = []
    for particle in particles:
        roots.append(
            dataclasses.replace(
                particle,
                ancestry=((particle.uid, particle.weight),),
                ancestry_variance_mass=particle.weight * particle.weight,
            )
        )
    validate_particles(roots)
    return roots


def generate_scenario(name: str, count: int, seed: int) -> List[Particle]:
    if count < 32:
        raise ValueError("synthetic scenarios require at least 32 roots")
    rng = random.Random(seed)
    particles: List[Particle] = []
    for index in range(count):
        if name == "correlated_core_halo":
            halo = rng.random() < 0.08
            scale_x = 2.4 if halo else 0.85
            scale_y = 1.8 if halo else 0.42
            g1 = rng.gauss(0.0, 1.0)
            g2 = rng.gauss(0.0, 1.0)
            x = scale_x * g1
            y = scale_y * (0.62 * g1 + math.sqrt(1.0 - 0.62**2) * g2)
            px = 0.22 * x + rng.gauss(0.0, 0.18)
            py = -0.31 * y + rng.gauss(0.0, 0.11)
            pz = 18.0 + 0.17 * x - 0.09 * y + rng.gauss(0.0, 0.35)
            weight = 1.0
        elif name == "weighted_bimodal_crossing":
            positive_lobe = rng.random() < 0.43
            centre = 1.45 if positive_lobe else -0.95
            x = centre + rng.gauss(0.0, 0.62 if positive_lobe else 0.78)
            y = (0.20 if positive_lobe else -0.13) + rng.gauss(0.0, 0.34)
            if rng.random() < 0.045:
                x += rng.choice((-1.0, 1.0)) * abs(rng.gauss(3.8, 0.9))
                y += rng.choice((-1.0, 1.0)) * abs(rng.gauss(2.5, 0.7))
            px = 0.38 * x + 0.07 * y + rng.gauss(0.0, 0.16)
            py = -0.12 * x - 0.27 * y + rng.gauss(0.0, 0.10)
            pz = 22.0 - 0.22 * x + rng.gauss(0.0, 0.42)
            weight = math.exp(rng.gauss(0.0, 0.55))
        else:
            raise ValueError(f"unknown synthetic scenario {name!r}")
        energy = physical_energy(px, py, pz)
        particles.append(Particle(index + 1, x, y, px, py, pz, energy, weight))
    return as_roots(particles)


def deposit_cic(particles: Sequence[Particle], grid: Grid) -> List[float]:
    grid.validate()
    density = [0.0] * (grid.nx * grid.ny)
    for particle in particles:
        ix, iy, fx, fy = grid.cic(particle.x, particle.y)
        fractions = (
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        )
        cells = ((ix, iy), (ix + 1, iy), (ix, iy + 1), (ix + 1, iy + 1))
        for fraction, (cx, cy) in zip(fractions, cells):
            density[cx * grid.ny + cy] += particle.weight * fraction
    return density


def exact_assignment_split(particles: Sequence[Particle], grid: Grid) -> Representation:
    validate_particles(particles)
    grid.validate()
    output: List[Particle] = []
    for particle in particles:
        ix, iy, fx, fy = grid.cic(particle.x, particle.y)
        fractions = (
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        )
        cells = ((ix, iy), (ix + 1, iy), (ix, iy + 1), (ix + 1, iy + 1))
        for ordinal, (fraction, cell) in enumerate(zip(fractions, cells)):
            if fraction <= 0.0:
                continue
            x, y = grid.centre(*cell)
            child_weight = particle.weight * fraction
            output.append(
                Particle(
                    uid=particle.uid * 16 + ordinal,
                    x=x,
                    y=y,
                    px=particle.px,
                    py=particle.py,
                    pz=particle.pz,
                    energy=particle.energy,
                    weight=child_weight,
                    cohort=particle.cohort,
                    ancestry=((particle.uid, child_weight),),
                    ancestry_variance_mass=(particle.weight * particle.weight * fraction),
                )
            )
    return Representation(
        "A_exact_assignment_preserving_CIC_split",
        output,
        exact_ancestry=True,
        details={
            "method": "one_cell_centred_atom_per_nonzero_CIC_coefficient",
            "new_independent_roots": 0,
        },
    )


def _weighted_scales(particles: Sequence[Particle]) -> Dict[str, float]:
    total = sum(p.weight for p in particles)
    result = {}
    for name in ("x", "y", "px", "py", "pz", "energy"):
        square = sum(p.weight * getattr(p, name) ** 2 for p in particles) / total
        result[name] = max(math.sqrt(square), 1.0e-12)
    return result


def _moment_features(particle: Particle, scales: Mapping[str, float]) -> List[float]:
    sx = scales["x"]
    sy = scales["y"]
    return [
        1.0,
        particle.x / sx,
        particle.y / sy,
        particle.x * particle.x / (sx * sx),
        particle.x * particle.y / (sx * sy),
        particle.y * particle.y / (sy * sy),
        particle.px / scales["px"],
        particle.py / scales["py"],
        particle.pz / scales["pz"],
        particle.energy / scales["energy"],
    ]


def _rref_null_vector(matrix: Sequence[Sequence[float]]) -> List[float]:
    if not matrix or not matrix[0]:
        raise AlgorithmRollback("empty constraint matrix")
    rows = len(matrix)
    columns = len(matrix[0])
    a = [list(row) for row in matrix]
    if any(len(row) != columns for row in a):
        raise AlgorithmRollback("ragged constraint matrix")
    maximum = max(abs(value) for row in a for value in row)
    tolerance = 1.0e-12 * max(1.0, maximum)
    pivot_columns: List[int] = []
    pivot_row = 0
    for column in range(columns):
        candidate = max(range(pivot_row, rows), key=lambda row: abs(a[row][column]))
        if abs(a[candidate][column]) <= tolerance:
            continue
        a[pivot_row], a[candidate] = a[candidate], a[pivot_row]
        pivot = a[pivot_row][column]
        a[pivot_row] = [value / pivot for value in a[pivot_row]]
        for row in range(rows):
            if row == pivot_row:
                continue
            factor = a[row][column]
            if abs(factor) <= tolerance:
                continue
            a[row] = [
                value - factor * pivot_value for value, pivot_value in zip(a[row], a[pivot_row])
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    free_columns = [column for column in range(columns) if column not in pivot_columns]
    if not free_columns:
        raise AlgorithmRollback("constraint subset has no numerical null direction")
    free = free_columns[-1]
    vector = [0.0] * columns
    vector[free] = 1.0
    for row in range(len(pivot_columns) - 1, -1, -1):
        column = pivot_columns[row]
        vector[column] = -sum(a[row][j] * vector[j] for j in free_columns)
    norm = max(abs(value) for value in vector)
    if norm <= tolerance:
        raise AlgorithmRollback("constraint null direction collapsed numerically")
    vector = [value / norm for value in vector]
    residual = max(
        abs(sum(row[column] * vector[column] for column in range(columns))) for row in matrix
    )
    if residual > 5.0e-9:
        raise AlgorithmRollback(f"constraint null residual is too large: {residual}")
    return vector


def _matrix_rank(matrix: Sequence[Sequence[float]]) -> int:
    if not matrix:
        return 0
    work = [list(row) for row in matrix]
    rows = len(work)
    columns = len(work[0])
    scale = max(abs(value) for row in work for value in row)
    tolerance = 1.0e-11 * max(1.0, scale)
    rank = 0
    for column in range(columns):
        candidate = max(range(rank, rows), key=lambda row: abs(work[row][column]))
        if abs(work[candidate][column]) <= tolerance:
            continue
        work[rank], work[candidate] = work[candidate], work[rank]
        pivot = work[rank][column]
        for row in range(rank + 1, rows):
            factor = work[row][column] / pivot
            for other in range(column, columns):
                work[row][other] -= factor * work[rank][other]
        rank += 1
        if rank == rows:
            break
    return rank


def constrained_moment_resample(particles: Sequence[Particle], target_count: int) -> Representation:
    """Positive Carathéodory elimination on a scaled ten-moment basis."""

    validate_particles(particles)
    scales = _weighted_scales(particles)
    features = [_moment_features(particle, scales) for particle in particles]
    constraint_matrix = [
        [features[column][row] for column in range(len(particles))]
        for row in range(len(features[0]))
    ]
    rank = _matrix_rank(constraint_matrix)
    if target_count < rank:
        raise AlgorithmRollback(f"target_count={target_count} is below constraint rank={rank}")
    if target_count >= len(particles):
        raise AlgorithmRollback("resampling target must reduce the ensemble")
    weights = [particle.weight for particle in particles]
    active = list(range(len(particles)))
    eliminations = 0
    while len(active) > target_count:
        active_matrix = [
            [constraint_matrix[row][column] for column in active]
            for row in range(len(constraint_matrix))
        ]
        direction = _rref_null_vector(active_matrix)
        positive = [
            weights[column] / value for column, value in zip(active, direction) if value > 1.0e-14
        ]
        if not positive:
            direction = [-value for value in direction]
            positive = [
                weights[column] / value
                for column, value in zip(active, direction)
                if value > 1.0e-14
            ]
        if not positive:
            raise AlgorithmRollback("null direction has no positive component")
        step = min(positive)
        next_active = []
        maximum_weight = max(weights[column] for column in active)
        for column, value in zip(active, direction):
            updated = weights[column] - step * value
            if updated < -2.0e-11 * max(1.0, maximum_weight):
                raise AlgorithmRollback("active-set update produced a negative weight")
            if abs(updated) <= 2.0e-13 * max(1.0, maximum_weight):
                weights[column] = 0.0
            else:
                weights[column] = updated
                next_active.append(column)
        if len(next_active) >= len(active):
            raise AlgorithmRollback("active-set iteration did not eliminate support")
        active = next_active
        eliminations += 1

    targets = [
        sum(particle.weight * feature[row] for particle, feature in zip(particles, features))
        for row in range(len(features[0]))
    ]
    residual = 0.0
    for row, target in enumerate(targets):
        achieved = sum(weights[column] * features[column][row] for column in active)
        residual = max(residual, abs(achieved - target) / max(1.0, abs(target)))
    if residual > 2.0e-8:
        raise AlgorithmRollback(f"moment-preserving active set failed closure: {residual}")
    output = [
        dataclasses.replace(
            particles[column],
            weight=weights[column],
            ancestry=((particles[column].uid, weights[column]),),
            ancestry_variance_mass=weights[column] * weights[column],
        )
        for column in active
    ]
    return Representation(
        "B_nonnegative_moment_constrained_resampling",
        output,
        exact_ancestry=False,
        details={
            "solver": "self_contained_Caratheodory_active_set",
            "feature_basis": [
                "1",
                "x",
                "y",
                "x2",
                "xy",
                "y2",
                "px",
                "py",
                "pz",
                "energy",
            ],
            "constraint_rank": rank,
            "eliminations": eliminations,
            "maximum_scaled_moment_residual": residual,
            "new_independent_roots": 0,
            "lineage_caveat": "quadrature reweighting has no exact positive many-root transport ledger",
        },
    )


def _perpendicular(vector: Tuple[float, float, float]) -> Tuple[float, float, float]:
    px, py, pz = vector
    magnitude = math.sqrt(px * px + py * py + pz * pz)
    if magnitude < 1.0e-15:
        return (1.0, 0.0, 0.0)
    components = (abs(px), abs(py), abs(pz))
    axis_index = min(range(3), key=lambda index: components[index])
    axis = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    axis[axis_index] = tuple(1.0 if i == axis_index else 0.0 for i in range(3))
    ax, ay, az = axis[axis_index]
    cross = (py * az - pz * ay, pz * ax - px * az, px * ay - py * ax)
    norm = math.sqrt(sum(value * value for value in cross))
    if norm < 1.0e-15:
        raise AlgorithmRollback("cannot construct deterministic perpendicular momentum")
    return tuple(value / norm for value in cross)  # type: ignore[return-value]


def vranic_two_output_merge(
    particles: Sequence[Particle], merge_grid: Grid, energy_scale: float = 1.0
) -> Representation:
    """Cell-local two-output conservative momentum/energy merge."""

    validate_particles(particles)
    merge_grid.validate()
    groups: Dict[Tuple[int, int], List[Particle]] = defaultdict(list)
    for particle in particles:
        ix, iy, _, _ = merge_grid.cic(particle.x, particle.y)
        groups[(ix, iy)].append(particle)
    output: List[Particle] = []
    merged_groups = 0
    for group_index, (cell, group) in enumerate(sorted(groups.items())):
        if len(group) <= 2:
            output.extend(group)
            continue
        total = sum(p.weight for p in group)
        xbar = sum(p.weight * p.x for p in group) / total
        ybar = sum(p.weight * p.y for p in group) / total
        pbar = tuple(
            sum(p.weight * getattr(p, component) for p in group) / total
            for component in ("px", "py", "pz")
        )
        ebar = energy_scale * sum(p.weight * p.energy for p in group) / total
        delta2 = ebar * ebar - MASS * MASS - sum(value * value for value in pbar)
        tolerance = 2.0e-12 * max(1.0, ebar * ebar)
        if delta2 < -tolerance:
            raise AlgorithmRollback(
                "two-output merge has no real on-shell solution for the requested energy"
            )
        delta = math.sqrt(max(0.0, delta2))
        direction = _perpendicular(pbar)
        ancestry_map: Dict[int, float] = defaultdict(float)
        root_square_mass = 0.0
        for particle in group:
            root_square_mass += particle.ancestry_variance_mass
            for root, contribution in particle.ancestry:
                ancestry_map[root] += contribution
        ancestry = tuple(
            (root, contribution * 0.5) for root, contribution in sorted(ancestry_map.items())
        )
        for sign in (-1.0, 1.0):
            momentum = tuple(value + sign * delta * axis for value, axis in zip(pbar, direction))
            on_shell = physical_energy(*momentum)
            if abs(on_shell - ebar) > 2.0e-10 * max(1.0, ebar):
                raise AlgorithmRollback("two-output merge failed its mass-shell closure")
            output.append(
                Particle(
                    uid=10_000_000 + 2 * group_index + (1 if sign > 0 else 0),
                    x=xbar,
                    y=ybar,
                    px=momentum[0],
                    py=momentum[1],
                    pz=momentum[2],
                    energy=ebar,
                    weight=0.5 * total,
                    cohort=group[0].cohort,
                    ancestry=ancestry,
                    ancestry_variance_mass=0.5 * root_square_mass,
                )
            )
        merged_groups += 1
    return Representation(
        "C_Vranic_style_two_output_merge",
        output,
        exact_ancestry=True,
        details={
            "method": "cell_local_equal_weight_two_output_on_shell_merge",
            "merged_groups": merged_groups,
            "position_rule": "both outputs_at_group_weighted_centroid",
            "conserves": ["charge", "position_centroid", "momentum_vector", "energy"],
            "does_not_conserve": ["position_covariance", "full_momentum_distribution"],
            "new_independent_roots": 0,
        },
    )


def _weighted_quantile(values: Sequence[Tuple[float, float]], quantile: float) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile outside [0,1]")
    ordered = sorted(values)
    total = sum(weight for _, weight in ordered)
    threshold = quantile * total
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def _residual_systematic_counts(
    group: Sequence[Particle], draws: int, rng: random.Random
) -> Dict[int, int]:
    total = sum(p.weight for p in group)
    expected = [draws * p.weight / total for p in group]
    counts = [math.floor(value) for value in expected]
    remainder = draws - sum(counts)
    residuals = [value - count for value, count in zip(expected, counts)]
    if remainder:
        residual_total = sum(residuals)
        if residual_total <= 0.0:
            raise AlgorithmRollback("residual sampler lost its positive remainder")
        start = rng.random() / remainder
        positions = [start + index / remainder for index in range(remainder)]
        cumulative = 0.0
        residual_index = 0
        normalized = [value / residual_total for value in residuals]
        for position in positions:
            while residual_index + 1 < len(normalized) and (
                cumulative + normalized[residual_index] <= position
            ):
                cumulative += normalized[residual_index]
                residual_index += 1
            counts[residual_index] += 1
    if sum(counts) != draws:
        raise AlgorithmRollback("residual sampler did not allocate all draws")
    return {p.uid: count for p, count in zip(group, counts) if count}


def stratified_residual_thinning(
    particles: Sequence[Particle], target_count: int, seed: int
) -> Representation:
    validate_particles(particles)
    if target_count < 3 or target_count >= len(particles):
        raise AlgorithmRollback("thinning target must be in [3,input_count)")
    moment = phase_space_moments(particles)
    radii = [(mahalanobis_radius(p, moment), p.weight) for p in particles]
    q80 = _weighted_quantile(radii, 0.80)
    q95 = _weighted_quantile(radii, 0.95)
    strata: List[List[Particle]] = [[], [], []]
    for particle in particles:
        radius = mahalanobis_radius(particle, moment)
        index = 0 if radius <= q80 else 1 if radius <= q95 else 2
        strata[index].append(particle)
    nonempty = [index for index, group in enumerate(strata) if group]
    if target_count < len(nonempty):
        raise AlgorithmRollback("thinning target cannot cover every nonempty stratum")
    masses = [sum(p.weight for p in group) for group in strata]
    allocation = [1 if group else 0 for group in strata]
    remaining = target_count - sum(allocation)
    mass_total = sum(masses)
    ideals = [remaining * mass / mass_total for mass in masses]
    floors = [math.floor(value) for value in ideals]
    allocation = [base + floor for base, floor in zip(allocation, floors)]
    left = target_count - sum(allocation)
    order = sorted(
        range(len(strata)),
        key=lambda index: (ideals[index] - floors[index], index),
        reverse=True,
    )
    for index in order[:left]:
        if strata[index]:
            allocation[index] += 1
    rng = random.Random(seed)
    by_uid = {particle.uid: particle for particle in particles}
    output: List[Particle] = []
    for stratum, (group, draws) in enumerate(zip(strata, allocation)):
        if not group:
            continue
        counts = _residual_systematic_counts(group, draws, rng)
        stratum_mass = masses[stratum]
        unit_weight = stratum_mass / draws
        for uid, count in sorted(counts.items()):
            particle = by_uid[uid]
            weight = count * unit_weight
            output.append(
                dataclasses.replace(
                    particle,
                    weight=weight,
                    ancestry=((particle.uid, weight),),
                    ancestry_variance_mass=weight * weight,
                )
            )
    return Representation(
        "D_experimental_stratified_residual_thinning",
        output,
        exact_ancestry=False,
        details={
            "status": "explicitly_experimental",
            "seed": seed,
            "stratum_quantiles": [0.80, 0.95],
            "stratum_input_counts": [len(group) for group in strata],
            "stratum_allocated_draws": allocation,
            "method": "mass_proportional_strata_plus_residual_systematic_sampling",
            "new_independent_roots": 0,
            "lineage_caveat": "selected roots are reweighted; duplicates are not independent",
        },
    )


def phase_space_moments(particles: Sequence[Particle]) -> Dict[str, object]:
    total = sum(p.weight for p in particles)
    mean_x = sum(p.weight * p.x for p in particles) / total
    mean_y = sum(p.weight * p.y for p in particles) / total
    cov_xx = sum(p.weight * (p.x - mean_x) ** 2 for p in particles) / total
    cov_xy = sum(p.weight * (p.x - mean_x) * (p.y - mean_y) for p in particles) / total
    cov_yy = sum(p.weight * (p.y - mean_y) ** 2 for p in particles) / total
    momentum = [sum(p.weight * getattr(p, name) for p in particles) for name in ("px", "py", "pz")]
    energy = sum(p.weight * p.energy for p in particles)
    return {
        "charge": total,
        "centroid": [mean_x, mean_y],
        "covariance": [[cov_xx, cov_xy], [cov_xy, cov_yy]],
        "momentum": momentum,
        "energy": energy,
    }


def mahalanobis_radius(particle: Particle, reference: Mapping[str, object]) -> float:
    mean_x, mean_y = reference["centroid"]  # type: ignore[misc]
    covariance = reference["covariance"]  # type: ignore[assignment]
    xx = covariance[0][0]
    xy = covariance[0][1]
    yy = covariance[1][1]
    determinant = xx * yy - xy * xy
    if determinant <= 1.0e-20:
        sx = math.sqrt(max(xx, 1.0e-20))
        sy = math.sqrt(max(yy, 1.0e-20))
        return math.hypot((particle.x - mean_x) / sx, (particle.y - mean_y) / sy)
    dx = particle.x - mean_x
    dy = particle.y - mean_y
    radius2 = (yy * dx * dx - 2.0 * xy * dx * dy + xx * dy * dy) / determinant
    return math.sqrt(max(0.0, radius2))


def _weighted_wasserstein_1d(
    first: Sequence[Tuple[float, float]], second: Sequence[Tuple[float, float]]
) -> float:
    total_first = sum(weight for _, weight in first)
    total_second = sum(weight for _, weight in second)
    events: Dict[float, List[float]] = defaultdict(lambda: [0.0, 0.0])
    for value, weight in first:
        events[value][0] += weight / total_first
    for value, weight in second:
        events[value][1] += weight / total_second
    coordinates = sorted(events)
    if len(coordinates) < 2:
        return 0.0
    cdf_first = 0.0
    cdf_second = 0.0
    distance = 0.0
    for index, coordinate in enumerate(coordinates[:-1]):
        cdf_first += events[coordinate][0]
        cdf_second += events[coordinate][1]
        distance += abs(cdf_first - cdf_second) * (coordinates[index + 1] - coordinate)
    return distance


def _grid_errors(reference: Sequence[float], candidate: Sequence[float]) -> Dict[str, float]:
    absolute = [abs(a - b) for a, b in zip(reference, candidate)]
    l1 = sum(absolute) / max(sum(abs(value) for value in reference), 1.0e-300)
    linf = max(absolute, default=0.0) / max(
        max((abs(value) for value in reference), default=0.0), 1.0e-300
    )
    return {"relative_l1": l1, "relative_linf": linf}


def _ancestry_metrics(
    representation: Representation, original: Sequence[Particle]
) -> Dict[str, object]:
    total = sum(p.weight for p in representation.particles)
    original_square = sum(p.weight * p.weight for p in original)
    original_support = total * total / original_square
    if representation.exact_ancestry:
        contributions: Dict[int, float] = defaultdict(float)
        variance_mass = 0.0
        for particle in representation.particles:
            variance_mass += particle.ancestry_variance_mass
            for root, contribution in particle.ancestry:
                contributions[root] += contribution
        input_by_id = {p.uid: p.weight for p in original}
        residual = max(
            (
                abs(contributions.get(root, 0.0) - weight) / max(abs(weight), 1.0e-300)
                for root, weight in input_by_id.items()
            ),
            default=0.0,
        )
        clone_support = total * total / variance_mass
        distinct = len([value for value in contributions.values() if value > 0.0])
        return {
            "exact_ancestry": True,
            "ancestry_max_relative_residual": residual,
            "distinct_source_roots": distinct,
            "clone_support_proxy": clone_support,
            "original_root_ess_ceiling": original_support,
            "manufactured_support": clone_support > original_support * (1.0 + 1.0e-10),
        }
    selected_weights: Dict[int, float] = defaultdict(float)
    for particle in representation.particles:
        selected_weights[particle.uid] += particle.weight
    square = sum(weight * weight for weight in selected_weights.values())
    proxy = total * total / square
    return {
        "exact_ancestry": False,
        "ancestry_max_relative_residual": None,
        "distinct_source_roots": len(selected_weights),
        "clone_support_proxy": min(proxy, original_support),
        "original_root_ess_ceiling": original_support,
        "manufactured_support": False,
        "interpretation": "conservative selected-support proxy; no exact positive many-root ledger",
    }


def compare_representation(
    original: Sequence[Particle], representation: Representation, grid: Grid
) -> Dict[str, object]:
    if not representation.committed:
        raise AlgorithmRollback("normal comparison received an uncommitted proposal")
    candidate = representation.particles
    validate_particles(candidate)
    reference_moments = phase_space_moments(original)
    candidate_moments = phase_space_moments(candidate)
    charge_reference = reference_moments["charge"]
    charge_candidate = candidate_moments["charge"]
    centroid_reference = reference_moments["centroid"]
    centroid_candidate = candidate_moments["centroid"]
    covariance_reference = reference_moments["covariance"]
    covariance_candidate = candidate_moments["covariance"]
    sx = math.sqrt(max(covariance_reference[0][0], 1.0e-300))
    sy = math.sqrt(max(covariance_reference[1][1], 1.0e-300))
    centroid_error = math.hypot(
        (centroid_candidate[0] - centroid_reference[0]) / sx,
        (centroid_candidate[1] - centroid_reference[1]) / sy,
    )
    covariance_difference = math.sqrt(
        sum(
            (covariance_candidate[i][j] - covariance_reference[i][j]) ** 2
            for i in range(2)
            for j in range(2)
        )
    )
    covariance_norm = math.sqrt(
        sum(covariance_reference[i][j] ** 2 for i in range(2) for j in range(2))
    )
    momentum_reference = reference_moments["momentum"]
    momentum_candidate = candidate_moments["momentum"]
    momentum_error = math.sqrt(
        sum((a - b) ** 2 for a, b in zip(momentum_candidate, momentum_reference))
    ) / max(math.sqrt(sum(value * value for value in momentum_reference)), 1.0e-300)
    energy_error = abs(candidate_moments["energy"] - reference_moments["energy"]) / max(
        abs(reference_moments["energy"]), 1.0e-300
    )
    reference_density = deposit_cic(original, grid)
    candidate_density = deposit_cic(candidate, grid)
    reference_radii = [
        (mahalanobis_radius(particle, reference_moments), particle.weight) for particle in original
    ]
    candidate_radii = [
        (mahalanobis_radius(particle, reference_moments), particle.weight) for particle in candidate
    ]
    q95 = _weighted_quantile(reference_radii, 0.95)
    q99 = _weighted_quantile(reference_radii, 0.99)

    def tail_mass(pairs: Sequence[Tuple[float, float]], threshold: float) -> float:
        return sum(weight for value, weight in pairs if value >= threshold) / sum(
            weight for _, weight in pairs
        )

    wasserstein_components = {}
    for name in ("x", "y", "px", "py"):
        scale = max(
            math.sqrt(sum(p.weight * getattr(p, name) ** 2 for p in original) / charge_reference),
            1.0e-12,
        )
        wasserstein_components[name] = (
            _weighted_wasserstein_1d(
                [(getattr(p, name), p.weight) for p in original],
                [(getattr(p, name), p.weight) for p in candidate],
            )
            / scale
        )
    wasserstein_components["mahalanobis_radius"] = _weighted_wasserstein_1d(
        reference_radii, candidate_radii
    )
    on_shell = max(
        abs(p.energy - physical_energy(p.px, p.py, p.pz)) / max(1.0, p.energy) for p in candidate
    )
    weight_sum = sum(p.weight for p in candidate)
    weight_square = sum(p.weight * p.weight for p in candidate)
    return {
        "input_count": len(original),
        "output_count": len(candidate),
        "compression_ratio_input_over_output": len(original) / len(candidate),
        "charge_relative_error": abs(charge_candidate - charge_reference)
        / max(abs(charge_reference), 1.0e-300),
        "centroid_normalized_error": centroid_error,
        "covariance_relative_frobenius_error": covariance_difference
        / max(covariance_norm, 1.0e-300),
        "momentum_relative_error": momentum_error,
        "energy_relative_error": energy_error,
        "maximum_relative_mass_shell_error": on_shell,
        "grid_projection": _grid_errors(reference_density, candidate_density),
        "normalized_wasserstein_1d": wasserstein_components,
        "mean_normalized_wasserstein_1d": statistics.mean(wasserstein_components.values()),
        "tail": {
            "reference_q95_radius": q95,
            "reference_q99_radius": q99,
            "reference_mass_at_or_above_q95": tail_mass(reference_radii, q95),
            "candidate_mass_at_or_above_q95": tail_mass(candidate_radii, q95),
            "reference_mass_at_or_above_q99": tail_mass(reference_radii, q99),
            "candidate_mass_at_or_above_q99": tail_mass(candidate_radii, q99),
        },
        "representative_weight_ess": weight_sum * weight_sum / weight_square,
        "ancestry": _ancestry_metrics(representation, original),
    }


def _representation_digest(representation: Representation) -> str:
    digest = hashlib.sha256()
    digest.update(representation.algorithm.encode("utf-8"))
    for particle in representation.particles:
        digest.update(str(particle.uid).encode("ascii"))
        for value in (
            particle.x,
            particle.y,
            particle.px,
            particle.py,
            particle.pz,
            particle.energy,
            particle.weight,
            particle.ancestry_variance_mass,
        ):
            digest.update(float(value).hex().encode("ascii"))
        for root, contribution in particle.ancestry:
            digest.update(str(root).encode("ascii"))
            digest.update(float(contribution).hex().encode("ascii"))
    return digest.hexdigest()


def benchmark(
    factory: Callable[[], Representation], repeats: int
) -> Tuple[Representation, Dict[str, object]]:
    if repeats < 1:
        raise ValueError("benchmark repeats must be positive")
    timings = []
    digests = []
    measured: Optional[Representation] = None
    peak_bytes = 0
    for repetition in range(repeats):
        gc.collect()
        if repetition == 0:
            tracemalloc.start()
        start = time.perf_counter_ns()
        result = factory()
        elapsed = time.perf_counter_ns() - start
        if repetition == 0:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            measured = result
        timings.append(elapsed / 1.0e6)
        digests.append(_representation_digest(result))
    if len(set(digests)) != 1:
        raise AlgorithmRollback("fixed-seed algorithm is not repeat deterministic")
    assert measured is not None
    return measured, {
        "repeats": repeats,
        "milliseconds": timings,
        "median_milliseconds": statistics.median(timings),
        "minimum_milliseconds": min(timings),
        "peak_tracemalloc_bytes": peak_bytes,
        "output_digest_sha256": digests[0],
        "repeat_deterministic": True,
    }


def _input_digest(particles: Sequence[Particle]) -> str:
    return _representation_digest(Representation("input", list(particles), True))


def run_failure_cases(particles: Sequence[Particle], grid: Grid) -> List[Dict[str, object]]:
    cases: List[Tuple[str, Callable[[], object]]] = [
        (
            "A_invalid_grid",
            lambda: exact_assignment_split(
                particles, Grid(1, grid.ny, grid.xmin, grid.xmax, grid.ymin, grid.ymax)
            ),
        ),
        (
            "B_target_below_constraint_rank",
            lambda: constrained_moment_resample(particles, 3),
        ),
        (
            "C_no_real_on_shell_solution",
            lambda: vranic_two_output_merge(particles, Grid(4, 4, -9, 9, -7, 7), 0.4),
        ),
        (
            "D_zero_target",
            lambda: stratified_residual_thinning(particles, 0, 991),
        ),
        (
            "negative_input_weight",
            lambda: exact_assignment_split(
                [dataclasses.replace(particles[0], weight=-1.0)] + list(particles[1:]),
                grid,
            ),
        ),
    ]
    baseline = _input_digest(particles)
    evidence = []
    for name, operation in cases:
        caught = None
        try:
            operation()
        except AlgorithmRollback as error:
            caught = str(error)
        evidence.append(
            {
                "case": name,
                "expected": "rollback",
                "observed": "rollback" if caught else "unexpected_success",
                "reason": caught,
                "input_unchanged": _input_digest(particles) == baseline,
                "pass": caught is not None and _input_digest(particles) == baseline,
            }
        )
    return evidence


def _flatten_row(scenario: str, result: Mapping[str, object]) -> Dict[str, object]:
    metrics = result["metrics"]
    performance = result["performance"]
    ancestry = metrics["ancestry"]
    tail = metrics["tail"]
    grid = metrics["grid_projection"]
    return {
        "scenario": scenario,
        "algorithm": result["algorithm"],
        "input_count": metrics["input_count"],
        "output_count": metrics["output_count"],
        "compression_ratio": metrics["compression_ratio_input_over_output"],
        "charge_relative_error": metrics["charge_relative_error"],
        "centroid_normalized_error": metrics["centroid_normalized_error"],
        "covariance_relative_error": metrics["covariance_relative_frobenius_error"],
        "momentum_relative_error": metrics["momentum_relative_error"],
        "energy_relative_error": metrics["energy_relative_error"],
        "grid_relative_l1": grid["relative_l1"],
        "grid_relative_linf": grid["relative_linf"],
        "mean_normalized_wasserstein": metrics["mean_normalized_wasserstein_1d"],
        "tail_q95_mass_absolute_error": abs(
            tail["candidate_mass_at_or_above_q95"] - tail["reference_mass_at_or_above_q95"]
        ),
        "tail_q99_mass_absolute_error": abs(
            tail["candidate_mass_at_or_above_q99"] - tail["reference_mass_at_or_above_q99"]
        ),
        "representative_weight_ess": metrics["representative_weight_ess"],
        "clone_support_proxy": ancestry["clone_support_proxy"],
        "original_root_ess_ceiling": ancestry["original_root_ess_ceiling"],
        "exact_ancestry": ancestry["exact_ancestry"],
        "manufactured_support": ancestry["manufactured_support"],
        "median_milliseconds": performance["median_milliseconds"],
        "peak_tracemalloc_bytes": performance["peak_tracemalloc_bytes"],
        "output_digest_sha256": performance["output_digest_sha256"],
    }


def run_comparison(count: int, repeats: int) -> Dict[str, object]:
    grid = Grid(40, 32, -10.0, 10.0, -7.5, 7.5)
    merge_grid = Grid(8, 8, -10.0, 10.0, -7.5, 7.5)
    scenarios = [
        ("correlated_core_halo", 2026090801),
        ("weighted_bimodal_crossing", 2026090802),
    ]
    report: Dict[str, object] = {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "standalone_not_production_integration": True,
        "mass_units": MASS,
        "configuration": {
            "particles_per_scenario": count,
            "benchmark_repeats": repeats,
            "diagnostic_grid": dataclasses.asdict(grid),
            "merge_grid": dataclasses.asdict(merge_grid),
            "moment_resampling_target_count": 16,
            "thinning_target_count": 64,
            "seeds": {name: seed for name, seed in scenarios},
            "thinning_seed_offset": 7000,
        },
        "environment": {
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "pid": os.getpid(),
            "linear_solver": "self_contained_active_set_no_numpy_scipy_required",
        },
        "scenarios": {},
        "failure_cases": [],
        "interpretation_guardrails": [
            "No split, merge, resample, or thinning output is a new independent root.",
            "Only algorithms A and C carry an exact positive many-root ancestry ledger.",
            "Grid projection is an observable distinct from phase-space distribution distance.",
            "Passing synthetic constraints is not FeeBL physics qualification.",
        ],
    }
    for scenario_name, seed in scenarios:
        original = generate_scenario(scenario_name, count, seed)
        factories: List[Callable[[], Representation]] = [
            lambda original=original: exact_assignment_split(original, grid),
            lambda original=original: constrained_moment_resample(original, 16),
            lambda original=original: vranic_two_output_merge(original, merge_grid),
            lambda original=original, seed=seed: stratified_residual_thinning(
                original, 64, seed + 7000
            ),
        ]
        results = []
        for factory in factories:
            representation, performance = benchmark(factory, repeats)
            results.append(
                {
                    "algorithm": representation.algorithm,
                    "status": "experimental_comparator_pass",
                    "exact_ancestry": representation.exact_ancestry,
                    "details": representation.details,
                    "performance": performance,
                    "metrics": compare_representation(original, representation, grid),
                }
            )
        report["scenarios"][scenario_name] = {
            "seed": seed,
            "input_digest_sha256": _input_digest(original),
            "input_root_count": len(original),
            "input_root_weight_ess": (
                sum(p.weight for p in original) ** 2 / sum(p.weight * p.weight for p in original)
            ),
            "results": results,
        }
        report["failure_cases"].extend(
            {"scenario": scenario_name, **case} for case in run_failure_cases(original, grid)
        )
    return report


def write_outputs(report: Mapping[str, object], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    json_path = output / "comparison.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = []
    for scenario, content in report["scenarios"].items():
        for result in content["results"]:
            rows.append(_flatten_row(scenario, result))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    failure_rows = list(report["failure_cases"])
    with (output / "failure_cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(failure_rows[0]))
        writer.writeheader()
        writer.writerows(failure_rows)


def print_summary(report: Mapping[str, object]) -> None:
    print(SCHEMA)
    print(
        "scenario algorithm out compression charge_err covariance_err grid_l1 "
        "mean_W1 clone_support median_ms peak_bytes"
    )
    for scenario, content in report["scenarios"].items():
        for result in content["results"]:
            row = _flatten_row(scenario, result)
            print(
                scenario,
                row["algorithm"],
                row["output_count"],
                f"{row['compression_ratio']:.6g}",
                f"{row['charge_relative_error']:.6g}",
                f"{row['covariance_relative_error']:.6g}",
                f"{row['grid_relative_l1']:.6g}",
                f"{row['mean_normalized_wasserstein']:.6g}",
                f"{row['clone_support_proxy']:.6g}",
                f"{row['median_milliseconds']:.6g}",
                row["peak_tracemalloc_bytes"],
            )
    failures = report["failure_cases"]
    passed = sum(1 for case in failures if case["pass"])
    print(f"rollback_cases={passed}/{len(failures)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=320)
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("--output must not already exist; evidence is never overwritten")
    report = run_comparison(arguments.particles, arguments.repeats)
    if not all(case["pass"] for case in report["failure_cases"]):
        raise SystemExit("one or more fail/rollback checks did not fail closed")
    write_outputs(report, arguments.output)
    print_summary(report)
    print(f"evidence={arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
