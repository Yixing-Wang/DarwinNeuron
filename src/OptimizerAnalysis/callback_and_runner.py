import time
import os
import numpy as np
import torch as T
from dataclasses import dataclass
from typing import Optional
from src.Utilities import evaluate_snn
from src.OptimizerAnalysis.pymoo_adapter import _unflatten_to_, _flatten_params

# Simple CSV writer for per-generation logs
@dataclass
class GenLogger:
    csv_path: str
    x_dir: Optional[str] = None          # where to save per-gen x
    epoch_x_dir: Optional[str] = None    # where to save per-epoch x

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if self.x_dir:
            os.makedirs(self.x_dir, exist_ok=True)
        if self.epoch_x_dir:
            os.makedirs(self.epoch_x_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", encoding="utf-8") as f:
                # minimal, file paths to x are recorded
                f.write("gen,n_evals,epoch,batch,best_f,best_acc,avg_acc,wall_time,epoch_acc_full,x_file,x_epoch_file\n")

    def write(self,
              gen: int, n_evals: int, epoch: int, batch: int,
              best_f: float, best_acc, avg_acc, wall_time: float,
              best_x: np.ndarray,
              is_epoch_end: bool,
              epoch_acc_full: Optional[float] = None):
        """Append one CSV row and save x files if dirs are configured."""
        ba = "" if best_acc is None else f"{best_acc}"
        aa = "" if avg_acc is None else f"{avg_acc}"

        # Save per-generation x
        x_file = ""
        if self.x_dir is not None and best_x is not None and best_x.size > 0:
            x_file = os.path.join(self.x_dir, f"best_x_gen{gen:04d}.npy")
            np.save(x_file, best_x.astype(np.float32))

        # Save per-epoch x (only at epoch end)
        x_epoch_file = ""
        if (
            is_epoch_end
            and self.epoch_x_dir is not None
            and best_x is not None
            and getattr(best_x, "size", 0) > 0
        ):
            x_epoch_file = os.path.join(
                self.epoch_x_dir, f"best_x_epoch{epoch:04d}.npy"
            )
            np.save(x_epoch_file, best_x.astype(np.float32))

        # with open(self.csv_path, "a", encoding="utf-8") as f:
        #     f.write(f"{gen},{n_evals},{epoch},{batch},{best_f},{ba},{aa},{wall_time},{x_file},{x_epoch_file}\n")
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{gen},{n_evals},{epoch},{batch},{best_f},{ba},{aa},{wall_time},"
                    f"{'' if epoch_acc_full is None else epoch_acc_full},"
                    f"{x_file},{x_epoch_file}\n")
            
class PymooDBCallback:
    def __init__(self, problem, gen_logger: GenLogger, print_line: bool = True):
        self.problem = problem
        self.gen_logger = gen_logger
        self.print_line = print_line
        self.t0 = time.time()
        self.last_epoch = None
        self.last_batch = None
        self.last_best_x = None
        self.last_best_file = None
        self.last_best_epoch_file = None

    def __call__(self, algorithm):
        res = algorithm.result()
        if res is None or res.F is None:
            return

        # count from 0
        n_gen_raw = int(getattr(algorithm, "n_gen", -1))
        gen_eff = max(0, n_gen_raw - 1)

        bpe = getattr(self.problem, "batches_per_epoch", None)
        if isinstance(bpe, int) and bpe > 0:
            epoch_idx    = gen_eff // bpe
            gen_in_epoch = gen_eff %  bpe
            is_epoch_end = (gen_in_epoch == bpe - 1)
        else:
            epoch_idx, gen_in_epoch, is_epoch_end = -1, gen_eff, False

        n_eval = int(getattr(getattr(algorithm, "evaluator", None), "n_eval", -1))
        best_f = float(res.F)

        try:
            x_best = np.array(res.X, dtype=np.float32).ravel()
        except Exception:
            x_best = np.array([], dtype=np.float32)

        # metrics
        best_acc = None
        try:
            best_acc = float(np.array(algorithm.opt.get("acc")).squeeze())
        except Exception:
            pass

        avg_acc = None
        try:
            pop_acc = algorithm.pop.get("acc")
            if pop_acc is not None:
                avg_acc = float(np.mean(np.asarray(pop_acc, dtype=np.float64)))
        except Exception:
            pass

        # at the end of epoch
        epoch_acc_full = None
        if is_epoch_end and x_best.size > 0 and getattr(self.problem, "eval_loader", None) is not None:
            model = self.problem.build_model().to(self.problem.device)
            vec = T.as_tensor(x_best, dtype=T.float32, device=self.problem.device)
            if self.problem.base_vec is not None:
                vec = self.problem.base_vec.to(self.problem.device) + vec
            _unflatten_to_(model, vec)

            stats_full = evaluate_snn(model, self.problem.eval_loader, self.problem.loss_fn, self.problem.device)
            epoch_acc_full = float(stats_full.correct / stats_full.total) if hasattr(stats_full, "total") else None

        wall = time.time() - self.t0

        self.gen_logger.write(
            gen=gen_eff,
            n_evals=n_eval,
            epoch=epoch_idx,
            batch=gen_in_epoch,       
            best_f=best_f,
            best_acc=best_acc,
            avg_acc=avg_acc,
            wall_time=wall,
            best_x=x_best,
            is_epoch_end=is_epoch_end,
            epoch_acc_full=epoch_acc_full,
        )

        self.last_epoch = epoch_idx
        self.last_batch = gen_in_epoch

        if self.print_line:
            ba_str = "NA" if best_acc is None else f"{best_acc:.4f}"
            aa_str = "NA" if avg_acc  is None else f"{avg_acc:.4f}"
            tag = " | epoch_end" if is_epoch_end else ""
            extra = "" if epoch_acc_full is None else f" | epoch_full_acc={epoch_acc_full:.4f}"
            print(f"[acc] gen={gen_eff:3d} | n_eval={n_eval:6d} | epoch={epoch_idx} | "
                  f"gen_in_epoch={gen_in_epoch} | best_f={best_f:.6f} | "
                  f"best_acc={ba_str} | avg_acc={aa_str}{tag}{extra}")