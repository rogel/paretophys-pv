"""Serial NSGA-II search with deterministic repair and within-run caching."""

import json
from pathlib import Path

import numpy as np

from .evolution import Individual, crossover_mutate, environmental_selection, random_genotype, tournament
from .search_space import repair_genotype
from .selection import OBJECTIVES, pareto_records
from .training import train_candidate


def run_search(data, output: str | Path, *, population=16, generations=8,
               method="nsga2", search_seed=3101, seed=2026, epochs=12,
               patience=3, batch_size=256, device="auto"):
    if population < 2 or generations < 1:
        raise ValueError("Use population >= 2 and generations >= 1")
    if method not in {"nsga2", "random"}:
        raise ValueError("Method must be nsga2 or random")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Search output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(search_seed)
    survivors, cache, requests = [], {}, []
    for generation in range(generations):
        children = []
        for slot in range(population):
            if generation == 0 or method == "random":
                raw = random_genotype(rng)
            else:
                raw = crossover_mutate(tournament(survivors, rng).genotype, tournament(survivors, rng).genotype, rng,
                                       crossover_probability=.9, mutation_probability=1 / 14)
            repaired = repair_genotype(raw)
            if repaired.constraint_violation > 1e-12:
                raise RuntimeError("Configuration remained infeasible after repair")
            genotype, key = repaired.genotype, repaired.genotype.key()
            hit = key in cache
            if not hit:
                cache[key] = train_candidate(data, genotype, output / "candidates" / key, epochs=epochs,
                                            patience=patience, batch_size=batch_size, seed=seed, device=device)
            record = cache[key]
            request = {"evaluation_id": len(requests) + 1, "generation": generation + 1, "slot": slot + 1,
                       "key": key, "cache_hit": hit, "repair_actions": repaired.actions, **{name: record[name] for name in OBJECTIVES}}
            requests.append(request)
            children.append(Individual(genotype, tuple(record[name] for name in OBJECTIVES), 0., request["evaluation_id"], raw.key(), raw != genotype, hit))
        survivors = environmental_selection(survivors + children, population)
        (output / "requests.json").write_text(json.dumps(requests, indent=2) + "\n")
        print(f"Generation {generation + 1}/{generations}: {len(requests)} requests, {len(cache)} trained configurations", flush=True)
    records = list(cache.values())
    front = pareto_records(records)
    (output / "evaluations.json").write_text(json.dumps(records, indent=2) + "\n")
    (output / "pareto_front.json").write_text(json.dumps(front, indent=2) + "\n")
    summary = {"method": method, "search_seed": search_seed, "training_seed": seed,
               "requests": len(requests), "unique_trainings": len(cache), "pareto_configurations": len(front),
               "population": population, "generations": generations, "epochs": epochs}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
