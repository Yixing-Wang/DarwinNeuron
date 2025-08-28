"""
Run comparison on a RANGE or LIST of NN problem ids with a SINGLE termination expressed in EPOCHS.
- If no --termination-id is given, we create a termination row with n_max_epochs=--epochs.
- The actual n_gen used at runtime is converted per-problem as: n_max_epochs * batches_per_epoch
  (requires TerminationRow.to_pymoo_termination_for_problem in method_comparison stack).
"""

import os
import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.OptimizerAnalysis.method_comparison import run_method_comparison, tune_algorithm_hyperparams
from src.OptimizerAnalysis.db_helpers import connect, ensure_tables, TerminationRow


def parse_ids(expr: str):
    expr = (expr or "").strip()
    out = []
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))

def _parse_int_list(expr: str):
    if not expr:
        return []
    return [int(x.strip()) for x in expr.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser(
        description="Run comparison on a RANGE or LIST of NN problem ids with a SINGLE termination expressed in EPOCHS."
    )
    ap.add_argument("--db", default="data/landscape-analysis.db")
    ap.add_argument("--ids", required=True, help="e.g. '0-8' or '1,3,5'")
    ap.add_argument("--out", default="src/OptimizerAnalysis/MethodComparisonRuns")
    ap.add_argument("--batch-size", type=int, default=516)
    ap.add_argument("--epochs", type=int, default=15,
                    help="How many EPOCHS to run; we store n_max_epochs in termination and convert to n_gen at runtime.")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma-separated seeds, e.g. '0,1,2'. If omitted, defaults to '0'.")
    ap.add_argument("--multi-run", action="store_true",
                    help="If set AND multiple seeds are given, run all seeds and save averaged logs per algorithm.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--save-history", action="store_true")
    ap.add_argument("--bounds-scale", type=float, default=5.0)
    ap.add_argument("--termination-id", type=int, default=None,
                    help="Reuse an existing termination_id; when absent, we create one with n_max_epochs=--epochs")
    ap.add_argument("--algorithm-ids", type=str, default=None,
                    help="Comma-separated algorithm ids to compare, e.g. '1,3'. If set, overrides inline defaults.")
    ap.add_argument("--tune-algo", type=str, default=None,
                    help="Algorithm name to tune before running (e.g., CMAES).")
    ap.add_argument("--tune-evals", type=int, default=50)
    args = ap.parse_args()

    # ---- parse seeds ----
    seeds = _parse_int_list(args.seeds) if args.seeds else [0]
    print(seeds)

    os.environ["LANDSCAPE_DB"] = args.db
    with connect(args.db) as con:
        ensure_tables(con)

    # shared termination (epoch-first)
    if args.termination_id is not None:
        term_id = int(args.termination_id)
        TerminationRow.lookup_by_id(term_id, db_path=args.db)
    else:
        term_row = TerminationRow(n_max_epochs=int(args.epochs))
        term_id = term_row.write_to_db(db_path=args.db)
    term_row = TerminationRow.lookup_by_id(term_id, db_path=args.db)

    # optional tuning
    tuned_algo_id = None
    if args.tune_algo:
        from src.OptimizerAnalysis.db_helpers import AlgorithmRow
        tune_ids = parse_ids(args.ids)
        if not tune_ids:
            raise ValueError("No valid --ids provided for tuning.")
        tune_problem_id = tune_ids[0]
        algo_obj, best = tune_algorithm_hyperparams(
            db_path=args.db,
            nn_problem_id=tune_problem_id,
            base_algo_name=args.tune_algo,
            base_algo_params={},
            term_row=term_row,
            batch_size=args.batch_size,
            bounds_scale=args.bounds_scale,
            seed=seeds[0],
            n_hpo_evals=args.tune_evals,
            verbose=args.verbose,
        )
        tuned_algo_id = AlgorithmRow(
            name=args.tune_algo.upper(),
            hyperparameter_problem=True,
            non_default_params=best
        ).write_to_db(args.db)

    algo_ids = _parse_int_list(args.algorithm_ids) if args.algorithm_ids else None
    if tuned_algo_id:
        algo_ids = [tuned_algo_id] if not algo_ids else (algo_ids + [tuned_algo_id])

    POP = 50
    default_methods = [
        {"name": "SGD",   "params": {"optimizer": "adam", "lr": 0.01}},
        {"name": "CMAES", "params": {"popsize": POP, "sigma": 0.3}},
        {"name": "DE",    "params": {"CR": 0.9, "F": 0.5, "pop_size": POP}},
        {"name": "PSO",   "params": {"pop_size": POP}},
        {"name": "GA",    "params": {"pop_size": POP}},
        {"name": "ES",    "params": {"pop_size": POP, "n_offsprings": 2*POP, "rule": 1.0/7.0}},
        {"name": "SRES",  "params": {"pop_size": POP, "n_offsprings": 2*POP, "rule": 1.0/7.0, "gamma": 0.85, "alpha": 0.2}},
        {"name": "ISRES", "params": {"pop_size": POP, "n_offsprings": 2*POP, "rule": 1.0/7.0, "gamma": 0.85, "alpha": 0.2}},
        # {"name": "NELDERMEAD",   "params": {}},
        # {"name": "PATTERNSEARCH","params": {}},
    ]

    all_runs = []
    ids = parse_ids(args.ids)
    for pid in ids:
        subout = Path(args.out) / f"NN_{pid}"
        subout.mkdir(parents=True, exist_ok=True)

        summaries = run_method_comparison(
            db_path=args.db,
            nn_problem_id=pid,
            methods=default_methods,
            termination=None,
            out_dir=str(subout),
            batch_size=args.batch_size,
            seeds=seeds,                
            multi_run=args.multi_run,     
            save_history=args.save_history,
            verbose=args.verbose,
            bounds_scale=args.bounds_scale,
            termination_id=term_id,
            algorithm_ids=algo_ids,
        )
        all_runs.append({"problem_id": pid, "termination_id": term_id, "summaries": summaries})

    batch_summary = Path(args.out) / "batch_summary.json"
    with open(batch_summary, "w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2)

    print(
        f"[batch] wrote {batch_summary}\n"
        f"[batch] termination_id={term_id} (epoch-based)\n"
        f"[batch] algos={algo_ids or 'defaults'}\n"
        f"[batch] seeds={seeds} | multi_run={args.multi_run}"
    )

if __name__ == "__main__":
    main()


# def parse_int_list(expr: str):
#     if not expr:
#         return []
#     return [int(x.strip()) for x in expr.split(",") if x.strip()]

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--db", default="data/landscape-analysis.db")
#     ap.add_argument("--ids", required=True, help="e.g. '0-8' or '1,3,5'")
#     ap.add_argument("--out", default="src/OptimizerAnalysis/MethodComparisonRuns")
#     ap.add_argument("--batch-size", type=int, default=516)
#     ap.add_argument("--epochs", type=int, default=15)  # epochs you want to run (full passes over dataset)
#     ap.add_argument("--seed", type=int, default=0)
#     ap.add_argument("--verbose", action="store_true")
#     ap.add_argument("--save-history", action="store_true")
#     ap.add_argument("--bounds-scale", type=float, default=5.0)
#     ap.add_argument("--termination-id", type=int, default=None,
#                     help="Reuse an existing termination_id; when absent, we create one with n_max_epochs=--epochs")
#     ap.add_argument("--algorithm-ids", type=str, default=None,
#                     help="Comma-separated algorithm ids to compare, e.g. '1,3'. If set, overrides inline defaults.")
#     ap.add_argument("--tune-algo", type=str, default=None,
#                     help="Algorithm name to tune before running (e.g., CMAES).")
#     ap.add_argument("--tune-evals", type=int, default=50)
#     args = ap.parse_args()

#     os.environ["LANDSCAPE_DB"] = args.db
#     with connect(args.db) as con:
#         ensure_tables(con)

#     # Resolve / create a shared termination row (epoch-first semantics).
#     # When creating, we store n_max_epochs=--epochs. At runtime, method_comparison should convert
#     # to ('n_gen', n_max_epochs * batches_per_epoch) per problem via TerminationRow.to_pymoo_termination_for_problem.
#     if args.termination_id is not None:
#         term_id = int(args.termination_id)
#         TerminationRow.lookup_by_id(term_id, db_path=args.db)  # validate presence
#     else:
#         term_row = TerminationRow(n_max_epochs=int(args.epochs))
#         term_id = term_row.write_to_db(db_path=args.db)

#     term_row = TerminationRow.lookup_by_id(term_id, db_path=args.db)

#     # (Optional) Hyperparameter tuning: writes best params as a new algorithm row to reuse later.
#     tuned_algo_id = None
#     if args.tune_algo:
#         tune_ids = parse_ids(args.ids)
#         if not tune_ids:
#             raise ValueError("No valid --ids provided for tuning.")
#         tune_problem_id = tune_ids[0]  # tune on the first requested problem

#         algo_obj, best = tune_algorithm_hyperparams(
#             db_path=args.db,
#             nn_problem_id=tune_problem_id,
#             base_algo_name=args.tune_algo,
#             base_algo_params={},           # start from defaults; your tune routine should handle search space
#             term_row=term_row,             # epoch-based termination; converted per-problem at runtime
#             batch_size=args.batch_size,
#             bounds_scale=args.bounds_scale,
#             seed=args.seed,
#             n_hpo_evals=args.tune_evals,
#             verbose=args.verbose,
#         )
#         # persist tuned params
#         from src.OptimizerAnalysis.db_helpers import AlgorithmRow
#         tuned_algo_id = AlgorithmRow(
#             name=args.tune_algo.upper(),
#             hyperparameter_problem=True,
#             non_default_params=best
#         ).write_to_db(args.db)

#     # Choose which algorithms to run:
#     algo_ids = parse_int_list(args.algorithm_ids) if args.algorithm_ids else None
#     if tuned_algo_id:
#         algo_ids = [tuned_algo_id] if not algo_ids else (algo_ids + [tuned_algo_id])

#     # Explicitly record per-algorithm population knobs in params (no common_pop indirection).
#     POP = 50
#     default_methods = [
#         # CMA-ES: popsize
#         {"name": "CMAES",       "params": {"popsize": POP, "sigma": 0.3}},

#         # DE/PSO/GA: pop_size
#         {"name": "DE",          "params": {"CR": 0.9, "F": 0.5, "pop_size": POP}},
#         {"name": "PSO",         "params": {"pop_size": POP}},
#         {"name": "GA",          "params": {"pop_size": POP}},

#         # ES/SRES/ISRES: parent pop_size & n_offsprings
#         {"name": "ES",          "params": {"pop_size": POP, "n_offsprings": 2*POP, "rule": 1.0/7.0}}, # this 2*pop is required by algorithm
#         {"name": "SRES",        "params": {"pop_size": POP, "n_offsprings": 2*POP,
#                                            "rule": 1.0/7.0, "gamma": 0.85, "alpha": 0.2}},
#         {"name": "ISRES",       "params": {"pop_size": POP, "n_offsprings": 2*POP,
#                                            "rule": 1.0/7.0, "gamma": 0.85, "alpha": 0.2}},

#         # Nelder-Mead / Pattern Search: no population knobs
#         {"name": "NELDERMEAD",  "params": {}},
#         {"name": "PATTERNSEARCH","params": {}},
#     ]

#     all_runs = []
#     ids = parse_ids(args.ids)
#     for pid in ids:
#         subout = Path(args.out) / f"NN_{pid}"
#         subout.mkdir(parents=True, exist_ok=True)

#         summaries = run_method_comparison(
#             db_path=args.db,
#             nn_problem_id=pid,
#             methods=default_methods,
#             termination=None,                 # not used because we pass termination_id
#             out_dir=str(subout),
#             batch_size=args.batch_size,
#             seeds=[args.seed],
#             multi_run=False,
#             save_history=args.save_history,
#             verbose=args.verbose,
#             bounds_scale=args.bounds_scale,
#             termination_id=term_id,           # shared across problems; converted per-problem at runtime
#             algorithm_ids=algo_ids,           # reuse algorithms by id if provided
#         )
#         all_runs.append({"problem_id": pid, "termination_id": term_id, "summaries": summaries})

#     batch_summary = Path(args.out) / "batch_summary.json"
#     with open(batch_summary, "w", encoding="utf-8") as f:
#         json.dump(all_runs, f, indent=2)

#     print(
#         f"[batch] wrote {batch_summary}\n"
#         f"[batch] termination_id={term_id} (epoch-based)\n"
#         f"[batch] algos={algo_ids or 'defaults'}"
#     )

# if __name__ == "__main__":
#     main()
