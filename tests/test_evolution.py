import numpy as np

from paretophys_pv.evolution import (
    GENE_DOMAINS,
    Individual,
    constraint_violation,
    crossover_mutate,
    environmental_selection,
    hypervolume,
    random_genotype,
    rank_and_crowding,
)


def individual(objectives, evaluation_id, violation=0.0) -> Individual:
    genotype = random_genotype(np.random.default_rng(evaluation_id))
    return Individual(
        genotype=genotype,
        objectives=tuple(objectives),
        constraint_violation=violation,
        evaluation_id=evaluation_id,
        raw_key=genotype.key(),
        repaired=False,
        cache_hit=False,
    )


def test_random_and_variation_stay_in_domains() -> None:
    rng = np.random.default_rng(3101)
    first = random_genotype(rng)
    second = random_genotype(rng)
    for _ in range(50):
        child = crossover_mutate(
            first,
            second,
            rng,
            crossover_probability=0.9,
            mutation_probability=1 / 14,
        )
        for name, domain in GENE_DOMAINS.items():
            assert getattr(child, name) in domain
        assert 1 <= child.state_mask <= 15


def test_constrained_selection_prefers_feasible_then_pareto() -> None:
    population = [
        individual((1.0, 3.0, 2.0), 1),
        individual((2.0, 2.0, 2.0), 2),
        individual((3.0, 1.0, 2.0), 3),
        individual((0.5, 0.5, 0.5), 4, violation=1.0),
        individual((4.0, 4.0, 4.0), 5),
    ]
    selected = environmental_selection(population, 3)
    assert {value.evaluation_id for value in selected} == {1, 2, 3}
    rank_and_crowding(selected)
    assert all(value.rank == 0 for value in selected)


def test_hypervolume_uses_feasible_nondominated_points() -> None:
    population = [
        individual((1.0, 3.0, 2.0), 1),
        individual((2.0, 2.0, 2.0), 2),
        individual((3.0, 1.0, 2.0), 3),
        individual((0.1, 0.1, 0.1), 4, violation=1.0),
    ]
    value = hypervolume(population, (4.0, 4.0, 4.0))
    assert value > 0
    assert value < 4.0**3


def test_penalty_constraint_detects_missing_future_state() -> None:
    rng = np.random.default_rng(12)
    genotype = random_genotype(rng)
    values = genotype.canonical_dict()
    values["state_mask"] = 0b1100
    invalid = type(genotype)(**values)
    assert constraint_violation(invalid) >= 1.0
