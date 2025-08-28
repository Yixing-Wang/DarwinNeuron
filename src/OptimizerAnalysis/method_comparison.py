"""
Fair comparison runner using the NEW DB schema (algorithm / termination / runs).

- Reuses a single termination_id across all algorithms for a problem.
- Logs per-generation CSV via PymooDBCallback (manually invoked).
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable

# ensure src import
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch as T
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.algorithms.hyperparameters import SingleObjectiveSingleRun, HyperparameterProblem
from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.core.parameters import set_params, hierarchical
from pymoo.optimize import minimize

from src.OptimizerAnalysis.pymoo_adapter import PymooSNNProblem
from src.OptimizerAnalysis.callback_and_runner import GenLogger, PymooDBCallback
from src.OptimizerAnalysis.db_helpers import (
    connect,
    ensure_tables,
    AlgorithmRow,
    TerminationRow,
    RunRow,
)
from src.LandscapeAnalysis.Pipeline import NNProblemConfig
from src.OptimizerAnalysis.sgd_wrapper import train_with_sgd

# ---------------- algorithm registry ----------------

ALGO_REGISTRY: Dict[str, Callable[[Dict], object]] = {}

def register_algorithm(key: str, factory: Callable[[Dict], object]):
    """Register an algorithm factory under an uppercase key."""
    ALGO_REGISTRY[key.upper()] = factory

# ---------------- factories ----------------

def _clamp_offsprings(p: Dict) -> Dict:
    q = dict(p or {})
    pop = q.get("pop_size", q.get("popsize"))
    if pop is not None:
        pop = int(pop)
        min_off = 2 * pop
        off = q.get("n_offsprings")
        if off is None or int(off) < min_off:
            q["n_offsprings"] = min_off
    return q

def _factory_cmaes(params: Dict) -> object:
    import numpy as np
    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
    p = dict(params or {})
    no_early = p.pop("no_early_stop", True)
    p.setdefault("normalize", False)
    if no_early:
        p.setdefault("tolx", 0.0)
        p.setdefault("tolfun", 0.0)
        p.setdefault("tolstagnation", np.inf)
    return CMAES(**p)

def _factory_de(params: Dict) -> object:
    from pymoo.algorithms.soo.nonconvex.de import DE
    return DE(**(params or {}))

def _factory_sgd(params: Dict) -> object:
    return train_with_sgd(**(params or {}))

# Optional algorithms (register if available in your pymoo version)
try:
    from pymoo.algorithms.soo.nonconvex.es import ES
    def _factory_es(params: Dict) -> object:
        return ES(**_clamp_offsprings(params))
    register_algorithm("ES", _factory_es)
except Exception:
    pass

try:
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    def _factory_pso(params: Dict) -> object:
        return PSO(**(params or {}))
    register_algorithm("PSO", _factory_pso)
except Exception:
    pass

try:
    from pymoo.algorithms.soo.nonconvex.ga import GA
    def _factory_ga(params: Dict) -> object:
        return GA(**(params or {}))
    register_algorithm("GA", _factory_ga)
except Exception:
    pass

try:
    from pymoo.algorithms.soo.nonconvex.nelder_mead import NelderMead
    def _factory_nelder_mead(params: Dict) -> object:
        return NelderMead(**(params or {}))
    register_algorithm("NELDERMEAD", _factory_nelder_mead)
except Exception:
    pass

try:
    from pymoo.algorithms.soo.nonconvex.pattern_search import PatternSearch
    def _factory_pattern_search(params: Dict) -> object:
        return PatternSearch(**(params or {}))
    register_algorithm("PATTERNSEARCH", _factory_pattern_search)
except Exception:
    pass

try:
    from pymoo.algorithms.soo.nonconvex.sres import SRES
    def _factory_sres(params: Dict) -> object:
        return SRES(**_clamp_offsprings(params))
    register_algorithm("SRES", _factory_sres)
except Exception:
    pass

try:
    from pymoo.algorithms.soo.nonconvex.isres import ISRES
    def _factory_isres(params: Dict) -> object:
        return ISRES(**_clamp_offsprings(params))
    register_algorithm("ISRES", _factory_isres)
except Exception:
    pass

# Always register base ones
register_algorithm("CMAES", _factory_cmaes)
register_algorithm("DE",    _factory_de)
register_algorithm("SGD",   _factory_sgd)

def _create_algorithm_object(name: str, params: Dict) -> object:
    """
    Create an algorithm object by name using the provided params.
    (No common_pop mapping here.)
    """
    key = (name or "CMAES").upper()
    factory = ALGO_REGISTRY.get(key)
    if factory is None:
        raise ValueError(f"Unknown algorithm: {key}. Registered: {sorted(ALGO_REGISTRY.keys())}")
    return factory(dict(params or {}))

# ---------------- helpers ----------------

def _extract_final_epoch_acc_full(csv_path: Path) -> Optional[float]:
    try:
        df = _read_gen_log_csv(csv_path)
        s = pd.to_numeric(df.get("epoch_acc_full", pd.Series(dtype=float)), errors="coerce").dropna()
        return None if s.empty else float(s.iloc[-1])
    except Exception:
        return None


def tune_algorithm_hyperparams(
    db_path: str,
    nn_problem_id: int,
    base_algo_name: str,
    base_algo_params: dict,
    term_row: TerminationRow,
    batch_size: int,
    bounds_scale: float,
    seed: int,
    n_hpo_evals: int = 50,
    verbose: bool = False,
):
    """under given problem_id and termination_id，search algorithm's hyperparameter，return (algo_obj, best_params_dict)"""
    device = "cuda" if T.cuda.is_available() else "cpu"
    problem, _, _ = _build_problem_and_batch(
        nn_problem_id, device=device,
        bounds_scale=bounds_scale, db_path=db_path, batch_size=batch_size
    )

    performance = SingleObjectiveSingleRun(
        problem,
        termination=term_row.to_pymoo_termination_for_problem(problem),
        seed=seed
    )

    algo = _create_algorithm_object(base_algo_name, dict(base_algo_params or {}))

    res = minimize(
        HyperparameterProblem(algo, performance),
        Optuna(),
        termination=('n_evals', int(n_hpo_evals)),
        seed=seed,
        verbose=verbose
    )

    # find best parameter after single run
    best = hierarchical(res.X)     # -> dict
    set_params(algo, best)
    return algo, best

def _loader_next_batch(loader: DataLoader):
    it = iter(loader)
    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)
    return next_batch

def _build_problem_and_batch(
    nn_problem_id: int,
    device: str,
    bounds_scale: float,
    db_path: str,
    batch_size: int = 516
):
    nn_cfg = NNProblemConfig.lookup_by_id(nn_problem_id, db_path=db_path)
    loader = nn_cfg.get_loader(db_path=db_path, batch_size=batch_size)
    build_model = lambda: nn_cfg.get_model(db_path=db_path)
    loss_fn = nn_cfg.get_loss(db_path)
    next_batch = _loader_next_batch(loader)

    problem = PymooSNNProblem(
        build_model=build_model,
        next_batch=next_batch,
        loss_fn=loss_fn,
        device=device,
        bounds_scale=float(bounds_scale),
        extra_metrics=True,
        batches_per_epoch=len(loader),   # <-- key: so problem returns epoch/batch
    )
    problem.eval_loader = loader
    return problem, next_batch, loader
    
def _resolve_termination_id(db_path: str, termination: Optional[Dict], termination_id: Optional[int]) -> int:
    if termination_id is not None:
        TerminationRow.lookup_by_id(termination_id, db_path=db_path)  # validate
        return int(termination_id)

    t = dict(termination or {})
    n_max_epochs = t.get("n_max_epochs")
    if n_max_epochs is None and t.get("type") in {"n_epoch", "n_epochs", "epoch", "epochs"}:
        n_max_epochs = t.get("value")

    row = TerminationRow(
        xtol=t.get("xtol"),
        cvtol=t.get("cvtol"),
        ftol=t.get("ftol"),
        period=t.get("period"),
        n_max_gen=t.get("n_max_gen") or (t.get("value") if t.get("type") == "n_gen" else None),
        n_max_evals=t.get("n_max_evals") or (t.get("value") if t.get("type") in {"n_eval","n_evals"} else None),
        time=t.get("time"),
        n_max_epochs=int(n_max_epochs) if n_max_epochs is not None else None,
    )
    return row.write_to_db(db_path=db_path)

def _apply_exact_termination(algo, problem, cb: PymooDBCallback, term_tuple,
                             seed: int = 0, verbose: bool = False, save_history: bool = True):
    """
    Use pymoo's OOP loop to enforce EXACTLY the same termination across algorithms.
    We pass the same termination tuple (from TerminationRow) into algorithm.setup().
    """
    # prepare the algorithm (same as minimize, but we drive the loop)
    algo.setup(problem, termination=term_tuple, seed=seed, verbose=verbose, save_history=save_history)

    # run until termination – this honors n_gen / n_eval / time consistently
    while algo.has_next():
        algo.next()
        cb(algo)

# ---------------- core run functions ----------------

def run_single(
    db_path: str,
    problem_type: str,
    problem_id: int,
    algorithm_name: Optional[str],
    algorithm_params: Optional[Dict],
    termination_id: int,
    out_dir: Path,
    batch_size: int,
    seeds: List[int],
    multi_run: bool,
    save_history: bool,
    verbose: bool,
    bounds_scale: float = 5.0,
    algorithm_id: Optional[int] = None,  # <— reuse an existing algorithm row if given
) -> Tuple[int, Dict]:
    """
    One run with given algorithm & fixed termination_id. (single-seed – repeat externally if multi_run)
    """

    device = "cuda" if T.cuda.is_available() else "cpu"

    # Resolve algorithm row:
    if algorithm_id is not None:
        # Load from DB and IGNORE provided name/params to ensure consistency
        arow = AlgorithmRow.lookup_by_id(int(algorithm_id), db_path=db_path)
        algorithm_name = arow.name
        algorithm_params = arow.non_default_params
        algo_id = int(algorithm_id)
    else:
        # Create/ensure row exists based on provided name/params
        arow = AlgorithmRow(
            name=algorithm_name or "CMAES",
            hyperparameter_problem=False,
            non_default_params=algorithm_params or {},
        )
        algo_id = arow.write_to_db(db_path)

    # termination row (reuse)
    term_row = TerminationRow.lookup_by_id(termination_id, db_path=db_path)

    algo_name_tag = (algorithm_name or "ALG").upper()
    log_dir = out_dir  # out_dir already points to <...>/<ALGO>
    log_dir.mkdir(parents=True, exist_ok=True)
    gen_log_path = str(log_dir / "gen_log.csv")
    result_path = str(log_dir / "result.json")
    x_save_dir   = str(log_dir / "best_x")        # per-generation .npy
    x_epoch_dir  = str(log_dir / "best_x_epoch")  # per-epoch .npy

    run_id = RunRow(
        problem_type=problem_type,
        problem_id=problem_id,
        algorithm_id=algo_id,
        termination_id=termination_id,
        batch_size=int(batch_size),
        seeds=list(seeds),
        multi_run=bool(multi_run),
        save_history=bool(save_history),
        verbose=bool(verbose),
        result_filename=result_path,
    ).write_to_db(db_path)

    # init GenLogger
    gen_logger = GenLogger(gen_log_path, x_dir=x_save_dir, epoch_x_dir=x_epoch_dir)

    if algorithm_name == 'SGD':
        assert problem_type=='nn', 'sgd is only supported for NN problems'
        term_tuple = ("n_gen",term_row.n_max_epochs)
        summary = train_with_sgd(
            # problem
            problem_id, batch_size, 
            # algorithm
            algorithm_params, term_tuple, 
            seed=(seeds[0] if seeds else 0),
            batch_logger=gen_logger,
            # others
            log_dir=log_dir, db_path=db_path)
        summary["algo"] = algo_name_tag
    
    else:  
        # Build problem
        problem, next_batch, _ = _build_problem_and_batch(
            problem_id if problem_type == "nn" else problem_id,
            device=device,
            bounds_scale=bounds_scale,
            db_path=db_path,
            batch_size=batch_size, 
        )
        # init term tuple
        term_tuple = term_row.to_pymoo_termination_for_problem(problem)
    
        # Algorithm setup
        algo = _create_algorithm_object(algorithm_name, algorithm_params or {})
        
        cb = PymooDBCallback(
            problem,
            gen_logger,
            print_line=True
        )

        # EXACT termination via oop loop:
        t_init = time.time()
        _apply_exact_termination(
            algo, problem, cb, term_tuple,
            seed=int(seeds[0]) if seeds else 0,
            verbose=verbose,
            save_history=save_history,
        )
        runtime = time.time() - t_init

        res = algo.result()
        
        x_best = cb.last_best_x
        if x_best is None:
            try:
                if getattr(res, "X", None) is not None:
                    x_best = np.asarray(res.X, dtype=np.float32).ravel()
            except Exception:
                x_best = None
        
        x_best_file = cb.last_best_file
        if x_best_file is None and x_best is not None and x_best.size > 0:
            x_best_file = str(out_dir / "x_best_final.npy")
            np.save(x_best_file, x_best.astype(np.float32))
        
        best_acc = None
        try:
            best_acc = float(np.array(algo.opt.get("acc")).squeeze())
        except Exception:
            pass
        
        summary = {
            "best_f": float(res.F),
            "best_acc": best_acc,
            "epoch_acc_full": _extract_final_epoch_acc_full(Path(gen_log_path)),
            "n_evals": int(getattr(getattr(algo, "evaluator", None), "n_eval", -1)),
            "n_gen": int(getattr(algo, "n_gen", -1)),
            "total_epoch": int(cb.last_epoch) if cb.last_epoch is not None else -1,
            "algo": algo_name_tag,
            "x_best_len": 0 if x_best is None else int(x_best.size),
            "x_best_file": x_best_file,
            "runtime": runtime,
        }
    
    # Save result file
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    RunRow.touch_result(run_id, db_path=db_path)

    print(f"[finish] problem_id={problem_id}, termination_id={termination_id}", flush=True)

    return run_id, summary

    
def _read_gen_log_csv(path: Path) -> pd.DataFrame:
    cols = ["gen","n_evals","epoch","batch","best_f","best_acc","avg_acc","wall_time","epoch_acc_full","x_file","x_epoch_file"]
    if not path.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    for c in ["gen","n_evals","epoch","batch","best_f","best_acc","avg_acc","wall_time","epoch_acc_full"]:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df[cols]

def _average_seed_logs(gen_logs: List[Path], out_csv: Path):
    dfs = [_read_gen_log_csv(p) for p in gen_logs if p.exists()]
    dfs = [d for d in dfs if not d.empty]
    if not dfs:
        return
    big = (pd.concat(dfs, keys=range(len(dfs)), names=["seed_idx","row"]).reset_index(level=0))
    agg = (big.groupby("gen", as_index=False)
              .agg(n_evals=("n_evals","max"),
                   epoch=("epoch","max"),
                   batch=("batch","max"),
                   best_f=("best_f","mean"),
                   best_acc=("best_acc","mean"),
                   avg_acc=("avg_acc","mean"),
                   wall_time=("wall_time","mean"),
                   epoch_acc_full=("epoch_acc_full","mean")))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)

def run_method_comparison(
    db_path: str,
    nn_problem_id: int,
    methods: List[Dict],
    termination: Optional[Dict] = None,
    out_dir: str = "src/OptimizerAnalysis/MethodComparisonRuns",
    batch_size: int = 516,
    seeds: Optional[List[int]] = None,
    multi_run: bool = False,
    save_history: bool = True,
    verbose: bool = False,
    bounds_scale: float = 5.0,
    termination_id: Optional[int] = None,
    algorithm_ids: Optional[List[int]] = None,
) -> List[Dict]:
    """
    Compare multiple algorithms on an NN problem using the same termination_id.
    Supports multi-run (multiple seeds) with averaged logs.
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds or [0])

    # shared termination
    term_id = _resolve_termination_id(db_path, termination=termination, termination_id=termination_id)

    # resolve algorithms
    if algorithm_ids:
        algo_specs = []
        for aid in algorithm_ids:
            row = AlgorithmRow.lookup_by_id(int(aid), db_path=db_path)
            algo_specs.append({"id": int(aid), "name": row.name, "params": row.non_default_params})
    else:
        algo_specs = [{"id": None,
                       "name": m.get("name") or m.get("method_name") or m.get("pymoo_algorithm") or "CMAES",
                       "params": m.get("params") or m.get("pymoo_params") or {}} for m in methods]

    all_summaries: List[Dict] = []

    for spec in algo_specs:
        algo_name  = spec["name"]
        algo_params = spec["params"]
        aid        = spec["id"]

        algo_dir = out / algo_name.upper()
        algo_dir.mkdir(parents=True, exist_ok=True)

        if multi_run and len(seeds) > 1:
            per_seed_summaries = []
            gen_logs: List[Path] = []

            for s in seeds:
                seed_dir = algo_dir / f"seed_{s}"
                seed_dir.mkdir(parents=True, exist_ok=True)

                run_id, summary = run_single(
                    db_path=db_path,
                    problem_type="nn",
                    problem_id=nn_problem_id,
                    algorithm_name=algo_name,
                    algorithm_params=algo_params,
                    termination_id=term_id,
                    out_dir=seed_dir,
                    batch_size=batch_size,
                    seeds=[s], 
                    multi_run=False, 
                    save_history=save_history,
                    verbose=verbose,
                    bounds_scale=bounds_scale,
                    algorithm_id=aid,
                )
                per_seed_summaries.append({"seed": s, **summary})
                gen_logs.append(seed_dir / "gen_log.csv")

            # averaged logs & final summary
            _average_seed_logs(gen_logs, algo_dir / "gen_log_avg.csv")

            seed_epoch_acc_finals = []
            for seed_summary, log_p in zip(per_seed_summaries, gen_logs):
                val = seed_summary.get("final_epoch_acc_full")
                if val is None:
                    val = _extract_final_epoch_acc_full(log_p)
                if val is not None:
                    seed_epoch_acc_finals.append(float(val))

            mean_final_epoch_acc_full = (float(np.mean(seed_epoch_acc_finals)) 
                                         if seed_epoch_acc_finals else None)

            # averaged summary
            best_f_mean = float(np.mean([r["best_f"] for r in per_seed_summaries if "best_f" in r]))
            best_acc_vals = [r.get("best_acc") for r in per_seed_summaries if r.get("best_acc") is not None]
            avg_summary = {
                "algo": algo_name.upper(),
                "termination_id": term_id,
                "seeds": seeds,
                "mean_best_f": best_f_mean,
                "mean_best_acc": (float(np.mean(best_acc_vals)) if best_acc_vals else None),
                "mean_n_evals": int(np.mean([r["n_evals"] for r in per_seed_summaries])),
                "mean_n_gen": int(np.mean([r["n_gen"]   for r in per_seed_summaries])),
                "mean_final_epoch_acc_full": mean_final_epoch_acc_full,
            }
            with open(algo_dir / "summary_avg.json", "w", encoding="utf-8") as f:
                json.dump(avg_summary, f, indent=2)

            all_summaries.append({
                "algo": algo_name.upper(),
                "termination_id": term_id,
                "seeds": seeds,
                "per_seed": per_seed_summaries,
                "average": avg_summary,
            })

        else:
            # single seed
            sub_out = algo_dir
            run_id, summary = run_single(
                db_path=db_path,
                problem_type="nn",
                problem_id=nn_problem_id,
                algorithm_name=algo_name,
                algorithm_params=algo_params,
                termination_id=term_id,
                out_dir=sub_out,
                batch_size=batch_size,
                seeds=seeds[:1],
                multi_run=False,
                save_history=save_history,
                verbose=verbose,
                bounds_scale=bounds_scale,
                algorithm_id=aid,
            )
            all_summaries.append({"run_id": run_id, "termination_id": term_id, **summary})

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    return all_summaries

# ---------------- CLI ----------------

def _parse_int_list(expr: Optional[str]) -> Optional[List[int]]:
    if not expr:
        return None
    return [int(x.strip()) for x in expr.split(",") if x.strip()]