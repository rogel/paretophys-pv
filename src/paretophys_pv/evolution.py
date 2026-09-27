"""Discrete mixed-variable evolutionary search operators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from pymoo.indicators.hv import HV

from paretophys_pv.search_space import (
    BACKBONE_VALUES,
    CORRECTION_CAP_VALUES,
    DROPOUT_VALUES,
    GATE_VALUES,
    Genotype,
    HIDDEN_VALUES,
    LAMBDA_R_VALUES,
    LAMBDA_S_VALUES,
    LAYER_VALUES,
    LEARNING_RATE_VALUES,
    LOOKBACK_VALUES,
    PATCH_VALUES,
    RESIDUAL_WIDTH_VALUES,
    STRIDE_VALUES,
    build_model,
    count_parameters,
)


GENE_DOMAINS: dict[str, tuple[Any, ...]] = {
    "lookback": LOOKBACK_VALUES,
    "backbone": BACKBONE_VALUES,
    "hidden_width": HIDDEN_VALUES,
    "layers": LAYER_VALUES,
    "dropout": DROPOUT_VALUES,
    "patch_length": PATCH_VALUES,
    "stride": STRIDE_VALUES,
    "state_mask": tuple(range(1, 16)),
    "residual_width": RESIDUAL_WIDTH_VALUES,
    "gate": GATE_VALUES,
    "correction_cap": CORRECTION_CAP_VALUES,
    "lambda_r": LAMBDA_R_VALUES,
    "lambda_s": LAMBDA_S_VALUES,
    "learning_rate": LEARNING_RATE_VALUES,
}
GENE_NAMES = tuple(GENE_DOMAINS)


@dataclass
class Individual:
    genotype: Genotype
    objectives: tuple[float, float, float]
    constraint_violation: float
    evaluation_id: int
    raw_key: str
    repaired: bool
    cache_hit: bool
    rank: int = 0
    crowding: float = 0.0

    @property
    def feasible(self) -> bool:
        return self.constraint_violation <= 1e-12


def random_genotype(rng: np.random.Generator) -> Genotype:
    values = {
        name: domain[int(rng.integers(0, len(domain)))]
        for name, domain in GENE_DOMAINS.items()
    }
    return Genotype(**values)


def crossover_mutate(
    first: Genotype,
    second: Genotype,
    rng: np.random.Generator,
    *,
    crossover_probability: float,
    mutation_probability: float,
) -> Genotype:
    left = first.canonical_dict()
    right = second.canonical_dict()
    child: dict[str, Any] = {}
    use_crossover = rng.random() < crossover_probability
    for name in GENE_NAMES:
        child[name] = (
            left[name]
            if not use_crossover or rng.random() < 0.5
            else right[name]
        )
        if rng.random() < mutation_probability:
            alternatives = [value for value in GENE_DOMAINS[name] if value != child[name]]
            child[name] = alternatives[int(rng.integers(0, len(alternatives)))]
    return Genotype(**child)


def constraint_violation(genotype: Genotype) -> float:
    violation = 0.0
    if genotype.state_mask & 0b0011 == 0:
        violation += 1.0
    if genotype.backbone == "compact_patchtst":
        violation += max(0.0, (genotype.patch_length - genotype.lookback) / genotype.lookback)
        violation += max(0.0, (genotype.stride - genotype.patch_length) / genotype.patch_length)
    parameters = count_parameters(build_model(genotype))
    violation += max(0.0, (parameters - 1_000_000) / 1_000_000)
    violation += max(0.0, (parameters * 4 - 10_000_000) / 10_000_000)
    violation += max(0.0, genotype.correction_cap - 0.20)
    return float(violation)


def constrained_dominates(first: Individual, second: Individual) -> bool:
    if first.feasible and not second.feasible:
        return True
    if second.feasible and not first.feasible:
        return False
    if not first.feasible and not second.feasible:
        return first.constraint_violation < second.constraint_violation - 1e-12
    a = np.asarray(first.objectives)
    b = np.asarray(second.objectives)
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_sort(population: Sequence[Individual]) -> list[list[int]]:
    dominates: list[list[int]] = [[] for _ in population]
    dominated_count = np.zeros(len(population), dtype=np.int64)
    fronts: list[list[int]] = [[]]
    for i, first in enumerate(population):
        for j, second in enumerate(population):
            if i == j:
                continue
            if constrained_dominates(first, second):
                dominates[i].append(j)
            elif constrained_dominates(second, first):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            population[i].rank = 0
            fronts[0].append(i)
    rank = 0
    while fronts[rank]:
        next_front: list[int] = []
        for i in fronts[rank]:
            for j in dominates[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    population[j].rank = rank + 1
                    next_front.append(j)
        rank += 1
        fronts.append(next_front)
    return fronts[:-1]


def assign_crowding(population: Sequence[Individual], front: Sequence[int]) -> None:
    if not front:
        return
    for index in front:
        population[index].crowding = 0.0
    if len(front) <= 2:
        for index in front:
            population[index].crowding = float("inf")
        return
    values = np.asarray([population[index].objectives for index in front], dtype=np.float64)
    for objective in range(values.shape[1]):
        order = np.argsort(values[:, objective], kind="mergesort")
        population[front[int(order[0])]].crowding = float("inf")
        population[front[int(order[-1])]].crowding = float("inf")
        span = values[order[-1], objective] - values[order[0], objective]
        if span <= 0:
            continue
        for position in range(1, len(order) - 1):
            index = front[int(order[position])]
            if np.isinf(population[index].crowding):
                continue
            previous_value = values[order[position - 1], objective]
            next_value = values[order[position + 1], objective]
            population[index].crowding += float((next_value - previous_value) / span)


def rank_and_crowding(population: Sequence[Individual]) -> list[list[int]]:
    fronts = nondominated_sort(population)
    for front in fronts:
        assign_crowding(population, front)
    return fronts


def environmental_selection(population: list[Individual], size: int) -> list[Individual]:
    fronts = rank_and_crowding(population)
    selected: list[Individual] = []
    for front in fronts:
        if len(selected) + len(front) <= size:
            selected.extend(population[index] for index in front)
            continue
        ordered = sorted(
            (population[index] for index in front),
            key=lambda individual: (-individual.crowding, individual.evaluation_id),
        )
        selected.extend(ordered[: size - len(selected)])
        break
    rank_and_crowding(selected)
    return selected


def tournament(population: Sequence[Individual], rng: np.random.Generator) -> Individual:
    indices = rng.choice(len(population), size=2, replace=False)
    first = population[int(indices[0])]
    second = population[int(indices[1])]
    if first.rank != second.rank:
        return first if first.rank < second.rank else second
    if first.crowding != second.crowding:
        return first if first.crowding > second.crowding else second
    return first if rng.random() < 0.5 else second


def feasible_nondominated(population: Sequence[Individual]) -> list[Individual]:
    feasible = [individual for individual in population if individual.feasible]
    if not feasible:
        return []
    fronts = nondominated_sort(feasible)
    return [feasible[index] for index in fronts[0]]


def hypervolume(population: Sequence[Individual], reference: Sequence[float]) -> float:
    front = feasible_nondominated(population)
    if not front:
        return 0.0
    objectives = np.asarray([individual.objectives for individual in front], dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    objectives = objectives[np.all(objectives < reference_array, axis=1)]
    if len(objectives) == 0:
        return 0.0
    return float(HV(ref_point=reference_array)(objectives))
