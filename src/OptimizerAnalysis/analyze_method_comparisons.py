import os, json, sqlite3
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ================== 可配置 ==================
DB = "data/landscape-analysis.db"
OUT_ROOT = Path("data/MethodComparisonRuns")   # run_batch_compare 的 out
PROBLEM_IDS: Iterable[int] = [0,1,2,3,4,5,6,7]       # 只分析这些 problem
BATCH_SIZE = 516
BOUNDS_SCALE = 5.0

# landscape 绘图参数
LANDSCAPE_GRID = 60            # 网格密度 (N x N)
LANDSCAPE_SPAN = 0.5           # 相对扰动幅度（以变量界或 sigma 尺度）；也可设绝对
LANDSCAPE_DIMS = (0, 1)        # 用哪两个维度做 2D 可视化

# ==================================================

# ------- 依赖你的工程内模块 -------
from src.LandscapeAnalysis.Pipeline import NNProblemConfig
from src.OptimizerAnalysis.pymoo_adapter import PymooSNNProblem
import torch as T
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

# ===== 固定算法颜色（跨所有图一致） =====
from matplotlib import cm

# 你要展示的算法清单（出现就固定色；没列到的会自动给一个可复现的颜色）
_ALG_COLOR_ORDER = [
    "CMAES", "DE", "PSO", "GA", "ES", "SRES", "ISRES",
    "NELDERMEAD", "PATTERNSEARCH", "SGD", "ADAM"
]
# 用 tab10 先给前 10 个；不够的话再从 viridis 里延展
_tab10 = plt.get_cmap("tab10").colors
_extra = [cm.viridis(i/10.0) for i in range(10)]
_palette = list(_tab10) + _extra

COLOR_BY_ALGO = {name: _palette[i % len(_palette)] for i, name in enumerate(_ALG_COLOR_ORDER)}

def _algo_color(name: str):
    key = (name or "").upper()
    if key in COLOR_BY_ALGO:
        return COLOR_BY_ALGO[key]
    # 没在预设清单里：给个可复现的颜色（按名称 hash 到 lerp 上）
    h = abs(hash(key)) % 997
    return cm.plasma((h % 997) / 996.0)


# ---------------- DB helpers ----------------

def _rows_for_problems(db: str, pids: Iterable[int]) -> List[Tuple]:
    """Fetch run rows (id, problem_type, problem_id, algo_name, result_filename) for given problem ids."""
    pids = list(pids)
    if not pids:
        return []
    placeholders = ",".join("?" for _ in pids)
    q = f"""
    SELECT r.id, r.problem_type, r.problem_id, a.name, r.result_filename
    FROM runs r
    JOIN algorithm a ON a.id = r.algorithm_id
    WHERE r.problem_type='nn' AND r.problem_id IN ({placeholders})
    ORDER BY r.problem_id, a.name, r.id;
    """
    with sqlite3.connect(db) as con:
        return con.execute(q, pids).fetchall()

# ---------------- IO helpers ----------------

def _read_gen_log(log_path: Path) -> pd.DataFrame:
    """
    Read per-generation CSV (old or new) and normalize columns.

    New logs columns (superset):
      gen,n_evals,epoch,batch,best_f,best_acc,avg_acc,wall_time,epoch_acc_full,x_file,x_epoch_file

    Old logs (examples):
      gen,n_evals,best_f,best_acc,avg_acc,wall_time
      or gen,n_evals,best_f,best_acc,wall_time
    """
    cols = ["gen","n_evals","epoch","batch","best_f","best_acc","wall_time","epoch_acc_full","x_file","x_epoch_file","avg_acc"]
    # 注意：我们最终不使用 avg_acc，但为了兼容旧 CSV 的列顺序，这里读进来后丢弃
    ordered_out = ["gen","n_evals","epoch","batch","best_f","best_acc","wall_time","epoch_acc_full","x_file","x_epoch_file"]

    if not log_path.exists():
        return pd.DataFrame(columns=ordered_out)

    df = pd.read_csv(log_path)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA

    # 规范数值列
    for c in ["gen","n_evals","epoch","batch","best_f","best_acc","wall_time","epoch_acc_full"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df[ordered_out]

def _read_result_json(result_path: Path) -> Dict:
    if result_path.exists():
        try:
            return json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

# -------- 扫描多 seed 结构并汇总 --------

def _gather_algo_runs(algo_dir: Path) -> Dict[str, List[Tuple[Optional[int], Path, Path]]]:
    """
    返回结构：
    {
      "seeds": [(seed, gen_log_csv_path, result_json_path), ...],   # 多 seed 情况
      "avg": Path or None                                           # gen_log_avg.csv 若存在
      "single": Path or None                                        # 单 seed 情况的 gen_log.csv
    }
    """
    out = {"seeds": [], "avg": None, "single": None}

    avg_csv = algo_dir / "gen_log_avg.csv"
    if avg_csv.exists():
        out["avg"] = avg_csv

    # 多 seed 目录
    for p in sorted(algo_dir.glob("seed_*")):
        try:
            seed = int(p.name.split("_", 1)[1])
        except Exception:
            continue
        gen_csv = p / "gen_log.csv"
        res_json = p / "result.json"
        if gen_csv.exists():
            out["seeds"].append((seed, gen_csv, res_json))

    # 单 seed 落地（无 seed_* 子目录）
    if not out["seeds"]:
        gen_csv = algo_dir / "gen_log.csv"
        if gen_csv.exists():
            out["single"] = gen_csv

    return out

def _load_all_runs(db: str, pids: Iterable[int], out_root: Path) -> pd.DataFrame:
    """
    加载并堆叠所有 run 的 gen 日志（支持多 seed）。
    返回 long-form: + seed 列（单 seed 用 0），+ 平均曲线另存一份 algorithm-level 也读入并标记 seed='avg'
    """
    rows = _rows_for_problems(db, pids)
    records = []
    for run_id, ptype, pid, algoname, result_file in rows:
        algo_name = algoname.upper()
        run_dir = Path(result_file).parent       # 可能是 .../ALGO 或 .../ALGO/seed_S
        algo_dir = run_dir if run_dir.name.upper() == algo_name else run_dir.parent

        layout = _gather_algo_runs(algo_dir)

        # 多 seed
        if layout["seeds"]:
            for seed, gen_csv, _res_json in layout["seeds"]:
                df = _read_gen_log(gen_csv)
                if df.empty:
                    continue
                df.insert(0, "seed", seed)
                df.insert(0, "algorithm", algo_name)
                df.insert(0, "problem_id", pid)
                df.insert(0, "run_id", run_id)
                records.append(df)
            # 加入平均曲线（若有）
            if layout["avg"] is not None and layout["avg"].exists():
                dfa = _read_gen_log(layout["avg"])
                if not dfa.empty:
                    dfa.insert(0, "seed", "avg")
                    dfa.insert(0, "algorithm", algo_name)
                    dfa.insert(0, "problem_id", pid)
                    dfa.insert(0, "run_id", run_id)
                    records.append(dfa)
        else:
            # 单 seed
            gen_csv = layout["single"]
            if gen_csv is not None and gen_csv.exists():
                df = _read_gen_log(gen_csv)
                if not df.empty:
                    df.insert(0, "seed", 0)
                    df.insert(0, "algorithm", algo_name)
                    df.insert(0, "problem_id", pid)
                    df.insert(0, "run_id", run_id)
                    records.append(df)

    if not records:
        return pd.DataFrame(columns=[
            "run_id","problem_id","algorithm","seed",
            "gen","n_evals","epoch","batch","best_f","best_acc","wall_time","epoch_acc_full","x_file","x_epoch_file"
        ])
    return pd.concat(records, ignore_index=True)

# ---------------- Plots ----------------

def per_problem_plots(df: pd.DataFrame, save_dir: Path):
    """
    per-problem:
      - best_f vs generation（按 algorithm, seed 分组；若包含 seed='avg' 则也画）
      - accuracy（使用 best_acc；不再使用 avg_acc）
      - 可选：在 epoch_end 处 scatter epoch_acc_full（如果有）
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    for pid, g in df.groupby("problem_id"):
        # best_f
        plt.figure()
        for (algo, seed), gg in g.groupby(["algorithm","seed"]):
            gg = gg.sort_values("gen")
            label = f"{algo}" if seed in [0, "avg", None] else f"{algo}/seed{seed}"
            if seed == "avg":
                label = f"{algo}/AVG"
            plt.plot(gg["gen"], gg["best_f"], label=label, color=_algo_color(algo))
        plt.xlabel("generation")
        plt.ylabel("best_f (lower=better)")
        plt.title(f"Problem {pid} – best_f vs generation (per seed)")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(save_dir / f"problem_{pid}_bestf_vs_gen.png", dpi=150)
        plt.close()

        # best_acc
        if g["best_acc"].notna().any():
            plt.figure()
            for (algo, seed), gg in g.groupby(["algorithm","seed"]):
                gg = gg.sort_values("gen")
                y = gg["best_acc"]
                label = f"{algo}" if seed in [0, "avg", None] else f"{algo}/seed{seed}"
                if seed == "avg":
                    label = f"{algo}/AVG"
                plt.plot(gg["gen"], y, label=label, color=_algo_color(algo))
            plt.xlabel("generation")
            plt.ylabel("best_acc")
            plt.title(f"Problem {pid} – best_acc vs generation (per seed)")
            plt.legend(ncol=2, fontsize=8)
            plt.tight_layout()
            plt.savefig(save_dir / f"problem_{pid}_acc_vs_gen.png", dpi=150)
            plt.close()

        # 在 epoch_end 处标注 epoch_acc_full（如果有）
        if g["epoch_acc_full"].notna().any():
            plt.figure()
            for (algo, seed), gg in g.groupby(["algorithm","seed"]):
                gg = gg.sort_values("gen")
                # 只在有值的地方画点（epoch_end）
                mask = gg["epoch_acc_full"].notna()
                if mask.any():
                    label = f"{algo}" if seed in [0, "avg", None] else f"{algo}/seed{seed}"
                    if seed == "avg":
                        label = f"{algo}/AVG"
                    plt.scatter(gg.loc[mask, "gen"], gg.loc[mask, "epoch_acc_full"], s=12, label=label, color=_algo_color(algo))
            plt.xlabel("generation (epoch end)")
            plt.ylabel("epoch_acc_full")
            plt.title(f"Problem {pid} – epoch_acc_full at epoch ends")
            plt.legend(ncol=2, fontsize=8)
            plt.tight_layout()
            plt.savefig(save_dir / f"problem_{pid}_epoch_full_acc_scatter.png", dpi=150)
            plt.close()

# ---------------- Tables / Leaderboards ----------------

def final_leaderboards(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    - 如果存在 seed='avg'，用 avg 曲线的最后一行作为该算法在该 problem 的代表；
      否则，对各 seed 的最后一行求均值，得到代表行。
    - 输出：
      - per_problem_table (算法×问题)：best_f / best_acc
      - wins：以 best_f 最小为胜
      - overall_means：跨问题均值（final_best_f / final_best_acc）
    """
    if df.empty:
        return {}

    # 每个 (problem, algorithm, seed) 的最后一代
    last = (df.sort_values(["problem_id","algorithm","seed","gen"])
              .groupby(["problem_id","algorithm","seed"])
              .tail(1))

    # 构建代表行
    reps = []
    for (pid, algo), block in last.groupby(["problem_id","algorithm"]):
        if (block["seed"] == "avg").any():
            row = block[block["seed"] == "avg"].iloc[-1]
            reps.append(row)
        else:
            row = block.iloc[-1].copy()
            row["best_f"]   = block["best_f"].mean()
            row["best_acc"] = block["best_acc"].mean()
            row["seed"] = "mean"
            reps.append(row)

    reps = pd.DataFrame(reps)

    # per-problem 表（不再包含 avg_acc）
    per_problem = reps.pivot_table(
        index="problem_id",
        columns="algorithm",
        values=["best_f","best_acc"],
        aggfunc="first"
    ).sort_index()

    # 胜场（best_f 越小越好）
    ranks = reps.copy()
    ranks["rank"] = ranks.groupby("problem_id")["best_f"].rank(method="min")
    wins = (ranks[ranks["rank"] == 1]
            .groupby("algorithm").size()
            .rename("wins").sort_values(ascending=False).to_frame())

    # 全局均值
    agg = (reps.groupby("algorithm")
           .agg(final_best_f=("best_f","mean"),
                final_best_acc=("best_acc","mean"),
                problems=("problem_id","nunique"))
           .sort_values(["final_best_f","final_best_acc"], ascending=[True, False]))

    return {"per_problem_table": per_problem, "wins": wins, "overall_means": agg}

# ---------------- Landscape 绘图 ----------------

def _build_problem(nn_problem_id: int, batch_size: int, db_path: str, bounds_scale: float):
    device = "cuda" if T.cuda.is_available() else "cpu"
    cfg = NNProblemConfig.lookup_by_id(nn_problem_id, db_path=db_path)
    loader = cfg.get_loader(db_path=db_path, batch_size=batch_size)
    build_model = lambda: cfg.get_model(db_path=db_path)
    loss_fn = CrossEntropyLoss()

    it = iter(loader)
    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    problem = PymooSNNProblem(
        build_model=build_model,
        next_batch=next_batch,
        loss_fn=loss_fn,
        device=device,
        bounds_scale=float(bounds_scale),
        extra_metrics=False,
        batches_per_epoch=len(loader),
    )
    return problem

def _eval_problem(problem: PymooSNNProblem, X: np.ndarray) -> np.ndarray:
    """对一批解 X 计算目标值 f；返回形状 (N,)"""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    out = np.zeros((X.shape[0], 1), dtype=np.float64)
    problem._evaluate(X, out, None)   # out[:,0] 填充为 f
    return out[:, 0]

def _plot_landscape_for_algo(
    problem_id: int,
    algo_dir: Path,
    analysis_dir: Path,
    dims: Tuple[int,int] = LANDSCAPE_DIMS,
    grid: int = LANDSCAPE_GRID,
    span: float = LANDSCAPE_SPAN,
    db_path: str = DB,
    batch_size: int = BATCH_SIZE,
    bounds_scale: float = BOUNDS_SCALE,
):
    """
    找最终 best-x（优先：算法目录下 seed 平均不存在 → 逐 seed，取该 seed 的 result.json 的 x_best_file；
    若缺失，用 gen_log.csv 的最后一行 x_file/x_epoch_file 兜底），然后在给定两维上做 2D 网格评估并绘图。
    """
    problem = _build_problem(problem_id, batch_size, db_path, bounds_scale)

    # 收集候选 best-x 文件
    x_files = []

    # 逐 seed
    for seed_dir in sorted(algo_dir.glob("seed_*")):
        rj = seed_dir / "result.json"
        if rj.exists():
            info = _read_result_json(rj)
            if info.get("x_best_file"):
                x_files.append(Path(info["x_best_file"]))
            else:
                gl = seed_dir / "gen_log.csv"
                df = _read_gen_log(gl)
                if not df.empty:
                    row = df.sort_values("gen").tail(1).iloc[0]
                    for col in ["x_epoch_file","x_file"]:
                        val = row.get(col)
                        if pd.notna(val) and str(val).strip():
                            x_files.append(Path(str(val).strip()))
                            break

    # 单 seed 情况
    if not x_files:
        rj = algo_dir / "result.json"
        if rj.exists():
            info = _read_result_json(rj)
            if info.get("x_best_file"):
                x_files.append(Path(info["x_best_file"]))
        if not x_files:
            gl = algo_dir / "gen_log.csv"
            df = _read_gen_log(gl)
            if not df.empty:
                row = df.sort_values("gen").tail(1).iloc[0]
                for col in ["x_epoch_file","x_file"]:
                    val = row.get(col)
                    if pd.notna(val) and str(val).strip():
                        x_files.append(Path(str(val).strip()))
                        break

    if not x_files:
        print(f"[landscape] {algo_dir} 没找到 x_best 文件，跳过。")
        return

    # 取第一个 best-x 文件作代表
    x_path = x_files[0]
    if not x_path.exists():
        maybe = algo_dir / x_path.name
        if maybe.exists():
            x_path = maybe
        else:
            print(f"[landscape] 找到的 x 文件不存在: {x_files[0]}")
            return

    x0 = np.load(x_path).astype(np.float32).ravel()
    n = x0.size
    i, j = dims
    if max(i,j) >= n:
        print(f"[landscape] 维度不足：x_dim={n}, dims={dims}")
        return

    # 变量界（若有）
    xl = getattr(problem, "xl", None)
    xu = getattr(problem, "xu", None)

    def _mk_axis(center, lo, hi):
        if lo is not None and hi is not None:
            half = span * float(hi - lo)
        else:
            half = span * max(1.0, abs(center))
        return np.linspace(center - half, center + half, grid)

    xi_axis = _mk_axis(x0[i], float(xl[i]) if xl is not None else None, float(xu[i]) if xu is not None else None)
    xj_axis = _mk_axis(x0[j], float(xl[j]) if xl is not None else None, float(xu[j]) if xu is not None else None)

    # 组装网格点
    XI, XJ = np.meshgrid(xi_axis, xj_axis, indexing="ij")
    X = np.tile(x0, (grid * grid, 1))
    X[:, i] = XI.ravel()
    X[:, j] = XJ.ravel()

    # 评估
    Z = _eval_problem(problem, X).reshape(grid, grid)

    # 画图
    algo_name = algo_dir.name.upper()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(Z.T, origin="lower",
                   extent=[xi_axis[0], xi_axis[-1], xj_axis[0], xj_axis[-1]],
                   aspect="auto")
    cs = ax.contour(XI, XJ, Z, levels=10, linewidths=0.6)
    ax.clabel(cs, inline=True, fontsize=7)
    ax.plot([x0[i]], [x0[j]], marker="o", ms=5)
    ax.set_xlabel(f"x[{i}]")
    ax.set_ylabel(f"x[{j}]")
    ax.set_title(f"Problem {problem_id} – {algo_name}\nLandscape around best-x")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    out_png = analysis_dir / f"problem_{problem_id}_{algo_name}_landscape_dim{i}-{j}.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def draw_all_landscapes(problems: Iterable[int], out_root: Path):
    for pid in problems:
        base = out_root / f"NN_{pid}"
        if not base.exists():
            continue
        analysis_dir = base / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        # 每个算法目录（上层可能是 seeds 子目录结构）
        for algo_dir in sorted([p for p in base.iterdir() if p.is_dir() and p.name.isupper()]):
            _plot_landscape_for_algo(
                problem_id=pid,
                algo_dir=algo_dir,
                analysis_dir=analysis_dir,
                dims=LANDSCAPE_DIMS,
                grid=LANDSCAPE_GRID,
                span=LANDSCAPE_SPAN,
                db_path=DB,
                batch_size=BATCH_SIZE,
                bounds_scale=BOUNDS_SCALE,
            )

# ===== 快速 landscape 评估 =====
FAST_LANDSCAPE = True
LANDSCAPE_TOPK_ALGOS = 3    # 每个 problem 只画 top-K 算法
LANDSCAPE_GRID = 31         # 小网格，快
USE_ONLY_AVG_SEED = True    # 优先 avg，其次 seed_0

def _pick_best_x_file_for_algo(algo_dir: Path) -> Optional[Path]:
    # 先 avg，再 seed_0，再其它 seed，再单文件
    candidates = []

    # avg 不直接给 x 文件，跳过；走到 seed_* 抓 result.json 或 gen_log 末行
    # 先 seed_0
    seed0 = algo_dir / "seed_0"
    if seed0.exists():
        rj = seed0 / "result.json"
        if rj.exists():
            info = _read_result_json(rj)
            if info.get("x_best_file"): return Path(info["x_best_file"])
        gl = seed0 / "gen_log.csv"
        if gl.exists():
            df = _read_gen_log(gl)
            if not df.empty:
                row = df.sort_values("gen").tail(1).iloc[0]
                for col in ["x_epoch_file","x_file"]:
                    val = row.get(col)
                    if pd.notna(val) and str(val).strip():
                        p = Path(str(val).strip())
                        if p.exists(): return p
                        alt = seed0 / p.name
                        if alt.exists(): return alt

    # 其它 seed
    for seed_dir in sorted(algo_dir.glob("seed_*")):
        if seed_dir.name == "seed_0": continue
        rj = seed_dir / "result.json"
        if rj.exists():
            info = _read_result_json(rj)
            if info.get("x_best_file"):
                p = Path(info["x_best_file"])
                if p.exists(): return p
                alt = seed_dir / p.name
                if alt.exists(): return alt
        gl = seed_dir / "gen_log.csv"
        if gl.exists():
            df = _read_gen_log(gl)
            if not df.empty:
                row = df.sort_values("gen").tail(1).iloc[0]
                for col in ["x_epoch_file","x_file"]:
                    val = row.get(col)
                    if pd.notna(val) and str(val).strip():
                        p = Path(str(val).strip())
                        if p.exists(): return p
                        alt = seed_dir / p.name
                        if alt.exists(): return alt

    # 单 seed 情况
    rj = algo_dir / "result.json"
    if rj.exists():
        info = _read_result_json(rj)
        if info.get("x_best_file"):
            p = Path(info["x_best_file"])
            if p.exists(): return p
            alt = algo_dir / p.name
            if alt.exists(): return alt
    gl = algo_dir / "gen_log.csv"
    if gl.exists():
        df = _read_gen_log(gl)
        if not df.empty:
            row = df.sort_values("gen").tail(1).iloc[0]
            for col in ["x_epoch_file","x_file"]:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    p = Path(str(val).strip())
                    if p.exists(): return p
                    alt = algo_dir / Path(str(val).strip()).name
                    if alt.exists(): return alt
    return None

def _random_orthonormal_2d(n_dim: int, rng: np.random.RandomState) -> np.ndarray:
    A = rng.randn(n_dim, 2)
    # Gram-Schmidt
    u1 = A[:, 0]; u1 = u1 / (np.linalg.norm(u1) + 1e-12)
    v  = A[:, 1] - np.dot(A[:, 1], u1) * u1
    u2 = v / (np.linalg.norm(v) + 1e-12)
    return np.stack([u1, u2], axis=1)   # [n_dim, 2]

def _plot_landscape_fast(
    problem_id: int,
    algo_dir: Path,
    analysis_dir: Path,
    db_path: str = DB,
    batch_size: int = BATCH_SIZE,
    bounds_scale: float = BOUNDS_SCALE,
    grid: int = LANDSCAPE_GRID,
    span: float = 0.5,
    seed_for_plane: int = 0,
):
    x_path = _pick_best_x_file_for_algo(algo_dir)
    if x_path is None:
        print(f"[landscape-fast] {algo_dir} 没找到 best-x，跳过。")
        return

    x0 = np.load(x_path).astype(np.float32).ravel()
    n = x0.size

    # 构建问题 & 单次 batch（复用）
    device = "cuda" if T.cuda.is_available() else "cpu"
    cfg = NNProblemConfig.lookup_by_id(problem_id, db_path=db_path)
    loader = cfg.get_loader(db_path=db_path, batch_size=batch_size)
    build_model = lambda: cfg.get_model(db_path=db_path).to(device)
    loss_fn = CrossEntropyLoss()

    it = iter(loader)
    try:
        xb, yb = next(it)
    except StopIteration:
        it = iter(loader); xb, yb = next(it)
    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

    # 构造二维随机正交方向
    rng = np.random.RandomState(seed_for_plane + 1314)
    U = _random_orthonormal_2d(n, rng)   # [n,2]

    # 坐标网格 & 平面点
    t = np.linspace(-span, +span, grid)
    T1, T2 = np.meshgrid(t, t, indexing="ij")
    # X = x0 + U[:,0]*T1 + U[:,1]*T2
    X = (x0[None, :] +
         T1.reshape(-1, 1) * U[:, 0][None, :] +
         T2.reshape(-1, 1) * U[:, 1][None, :]).astype(np.float32)   # [G^2, n]

    # 复用一个模型实例，在 no_grad 下逐点替换参数并前向
    model = build_model()
    with T.no_grad():
        Z = np.empty((X.shape[0],), dtype=np.float64)
        y_long = yb.long() if yb.dtype != T.long else yb
        for i in range(X.shape[0]):
            vec = T.from_numpy(X[i]).to(device)
            # 写入参数
            idx = 0
            for p in model.parameters():
                n_i = p.numel()
                p.copy_(vec[idx:idx+n_i].view_as(p))
                idx += n_i
            # forward (可直接调用你的 run_snn_on_batch 简化)
            out = model(xb)
            # 取 logits：假设 out=(spikes, mem) or [T,B,C] / [B,C]
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                mem = out[1]
                logits = mem[-1] if mem.dim() == 3 else mem
            else:
                logits = out if out.dim() == 2 else out[-1]
            loss = loss_fn(logits, y_long)
            Z[i] = float(loss.item())
            # 清理中间（可选）
            del out, logits, loss
        Z = Z.reshape(grid, grid)

    # 绘图
    algo_name = algo_dir.name.upper()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(Z.T, origin="lower",
                   extent=[-span, span, -span, span], aspect="auto")
    cs = ax.contour(T1, T2, Z, levels=10, linewidths=0.6)
    ax.clabel(cs, inline=True, fontsize=7)
    ax.plot([0], [0], marker="o", ms=5, color=_algo_color(algo_name))  # 中心点
    ax.set_xlabel("u1"); ax.set_ylabel("u2")
    ax.set_title(f"Problem {problem_id} – {algo_name}\nFast landscape around best-x (random 2D)")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    out_png = analysis_dir / f"problem_{problem_id}_{algo_name}_landscape_fast.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def _pick_topk_algorithms_for_problem(df_long: pd.DataFrame, pid: int, k: int) -> List[str]:
    # 取该 problem 的代表成绩（seed='avg' 优先，否则各 seed 的最后一代求均值）
    g = df_long[df_long["problem_id"] == pid]
    if g.empty: return []
    last = (g.sort_values(["algorithm","seed","gen"])
              .groupby(["algorithm","seed"]).tail(1))
    reps = []
    for algo, block in last.groupby("algorithm"):
        if (block["seed"] == "avg").any():
            row = block[block["seed"] == "avg"].iloc[-1]
            reps.append((algo, float(row["best_f"])))
        else:
            reps.append((algo, float(block["best_f"].mean())))
    reps.sort(key=lambda x: x[1])  # best_f 越小越好
    return [a for a, _ in reps[:k]]

def draw_all_landscapes_fast(problems: Iterable[int], out_root: Path, df_long: pd.DataFrame):
    for pid in problems:
        base = out_root / f"NN_{pid}"
        if not base.exists(): 
            continue
        analysis_dir = base / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        algos_topk = _pick_topk_algorithms_for_problem(df_long, pid, LANDSCAPE_TOPK_ALGOS)
        if not algos_topk:
            print(f"[landscape-fast] problem {pid} 无可用算法，跳过。")
            continue

        for algo_dir in sorted([p for p in base.iterdir() if p.is_dir() and p.name.upper() in algos_topk]):
            _plot_landscape_fast(
                problem_id=pid,
                algo_dir=algo_dir,
                analysis_dir=analysis_dir,
                db_path=DB,
                batch_size=BATCH_SIZE,
                bounds_scale=BOUNDS_SCALE,
                grid=LANDSCAPE_GRID,
                span=0.5,
                seed_for_plane=0,
            )

# ---------------- Main ----------------

def main():
    df = _load_all_runs(DB, PROBLEM_IDS, OUT_ROOT)
    if df.empty:
        print("No gen logs found. Did the runs write result.json & gen_log.csv?")
        return

    for pid in PROBLEM_IDS:
        (OUT_ROOT / f"NN_{pid}" / "analysis").mkdir(parents=True, exist_ok=True)

    # 保存各问题长表
    for pid, g in df.groupby("problem_id"):
        analysis_dir = OUT_ROOT / f"NN_{pid}" / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        g.to_csv(analysis_dir / "all_runs_long.csv", index=False)

    # 趋势图（颜色固定）
    for pid, g in df.groupby("problem_id"):
        per_problem_plots(g, OUT_ROOT / f"NN_{pid}" / "analysis")

    # 榜单
    tabs = final_leaderboards(df)
    if tabs:
        global_dir = OUT_ROOT / "analysis"
        global_dir.mkdir(parents=True, exist_ok=True)
        for name, table in tabs.items():
            print(f"\n=== {name} ===")
            try:
                print(table.round(6))
            except Exception:
                print(table)
            (global_dir / f"{name}.csv").write_text(table.to_csv())

    # Landscape — 快模式 or 关闭
    if FAST_LANDSCAPE:
        draw_all_landscapes_fast(PROBLEM_IDS, OUT_ROOT, df)
    else:
        print("[landscape] FAST_LANDSCAPE=False，跳过。")

if __name__ == "__main__":
    main()
