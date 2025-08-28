from typing import Any, Optional, Tuple

import numpy as np
import torch as T
import torch.nn as nn
from pymoo.core.problem import Problem
from src.Utilities import run_snn_on_batch, evaluate_snn
from torch.nn.utils import parameters_to_vector

# ---------------- small helpers ----------------

def _flatten_params(model: nn.Module) -> T.Tensor:
    return T.cat([p.detach().reshape(-1) for p in model.parameters()])

def _unflatten_to_(model: nn.Module, vec: T.Tensor) -> None:
    i = 0
    with T.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vec[i:i+n].view_as(p))
            i += n

def _to_2d_float_array(X: Any, expected_nvar: Optional[int] = None) -> np.ndarray:
    X = np.asarray(X)
    if X.dtype == object:
        X = np.stack([np.asarray(row).astype(np.float32).ravel() for row in X], axis=0)
    X = X.astype(np.float32, copy=False)
    if X.ndim == 1:
        X = X[None, :]
    if expected_nvar is not None and X.shape[1] != expected_nvar:
        raise RuntimeError(f"[pymoo_adapter] Expected n_var={expected_nvar}, but got row length {X.shape[1]}")
    return X

# ---------------- Problem ----------------

class PymooSNNProblem(Problem):
    """
    build_model(): -> fresh nn.Module
    next_batch():  -> (xb, yb)
    eval_loader:   -> DataLoader for full-dataset evaluation (set from outside)
    """

    def __init__(self,
                 build_model,
                 next_batch,
                 loss_fn: Optional[nn.Module] = None,
                 device: str = "cpu",
                 bounds_scale: float = 5.0,
                 extra_metrics: bool = True,
                 base_vector: Optional[np.ndarray] = None,
                 batches_per_epoch: Optional[int] = 3):
        self.build_model = build_model
        self.next_batch = next_batch
        self.device = device
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.extra_metrics = extra_metrics

        template = self.build_model().to(self.device)
        with T.no_grad():
            flat = _flatten_params(template)
        n_var = flat.numel()

        self.base_vec = None
        if base_vector is not None:
            bv = np.asarray(base_vector, dtype=np.float32).ravel()
            if bv.shape[0] != n_var:
                raise ValueError(f"base_vector length {bv.shape[0]} != n_var {n_var}")
            self.base_vec = T.from_numpy(bv)

        xl = -bounds_scale * np.ones(n_var, dtype=np.float32)
        xu = +bounds_scale * np.ones(n_var, dtype=np.float32)
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu, elementwise_evaluation=False)

        self.batches_per_epoch = int(batches_per_epoch) if batches_per_epoch else None
        self.eval_loader = None  # DataLoader

    @T.no_grad()
    def _evaluate(self, X, out, *args, **kwargs):
        X = _to_2d_float_array(X, expected_nvar=self.n_var)
        pop = X.shape[0]

        xb, yb = self.next_batch()
        xb = xb.to(self.device, non_blocking=True)
        yb = yb.to(self.device, non_blocking=True)

        F   = np.zeros((pop, 1), dtype=np.float64)
        ACC = np.zeros((pop,),   dtype=np.float64) if self.extra_metrics else None

        for i in range(pop):
            vec = T.as_tensor(X[i], dtype=T.float32, device=self.device)
            if self.base_vec is not None:
                vec = self.base_vec.to(self.device) + vec

            model = self.build_model().to(self.device)
            _unflatten_to_(model, vec)

            stats = run_snn_on_batch(model, xb, yb, self.loss_fn)
            F[i, 0] = float(stats.loss.item() if hasattr(stats.loss, "item") else stats.loss)
            if self.extra_metrics:
                ACC[i] = float(stats.get_accuracy())

        out["F"] = F
        if self.extra_metrics:
            out["acc"] = ACC