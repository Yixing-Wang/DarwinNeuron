import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from typing import Callable, Generator, List, Optional

from .EvolutionAlgorithm import EA
from ..Training import SNNStats

# -----------------------------------------------------------------------------
# Pooling‑Pseudo‑PSO base model (PPSOModel)
# -----------------------------------------------------------------------------
class PPSOModel(EA):
    """PSO‑style optimiser compatible with new *stats* interface.

    `update(model_stats_fn)` now expects a function that returns a **SNNStats**
    instance for each candidate model. Loss and accuracy are extracted from that
    object, so the training loop only builds *one* forward pass per sample.
    """

    # ---------------- initialisation ----------------
    def __init__(
        self,
        Model: nn.Module,
        *model_args,
        sample_size: int = 32,
        param_std: float = 0.05,
        lr: float = 0.1,
        inertia: float = 0.5,
        c1: float = 1.5,
        c2: float = 1.5,
        mirror: bool = True,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.ModelCls = Model
        self.device = torch.device(device)
        super().__init__(Model, device)

        # hyper‑params
        self.model_args = model_args
        self.sample_size = (sample_size // 2) * 2 if mirror else sample_size
        self.std = param_std
        self.lr = lr
        self.w, self.c1, self.c2 = inertia, c1, c2
        self.mirror = mirror

        # init centre vector
        tmp = Model(*model_args).to(self.device)
        with torch.no_grad():
            self.mean = parameters_to_vector(tmp.parameters()).detach()
        self.dim = self.mean.numel()

        # PSO state
        self.velocity = torch.zeros_like(self.mean)
        self.group_best = self.mean.clone()
        self.history_best = self.mean.clone()
        self.history_best_loss = torch.tensor(float("inf"), device=self.device)

        # store last stats fn for _loss_of
        self._last_stats_fn: Optional[Callable[[nn.Module], SNNStats]] = None

    # -------------- helpers --------------
    def _flat_to_model(self, flat: torch.Tensor) -> nn.Module:
        m = self.ModelCls(*self.model_args).to(self.device)
        with torch.no_grad():
            vector_to_parameters(flat, m.parameters())
        return m

    # -------------- sampling --------------
    def samples(self) -> Generator[nn.Module, None, None]:
        half = self.sample_size // 2 if self.mirror else self.sample_size
        eps = torch.randn(half, self.dim, device=self.device)
        if self.mirror:
            eps = torch.cat([eps, -eps], dim=0)
        self._last_eps = eps
        flats = self.mean.unsqueeze(0) + self.std * eps
        for fl in flats:
            yield self._flat_to_model(fl)

    # -------------- PSO update core --------------
    def _pso_step(self, losses: torch.Tensor) -> None:
        flats = self.mean.unsqueeze(0) + self.std * self._last_eps
        best_idx = torch.argmin(losses)
        best_flat = flats[best_idx]
        best_loss = losses[best_idx]

        if best_loss < self.history_best_loss:
            self.history_best_loss = best_loss
            self.history_best = best_flat.clone()

        centre_loss = losses.mean()
        if centre_loss < self._loss_of(self.group_best):
            self.group_best = self.mean.clone()

        r1, r2 = torch.rand(2, device=self.device)
        self.velocity = (
            self.w * self.velocity
            + self.c1 * r1 * (self.group_best - self.mean)
            + self.c2 * r2 * (self.history_best - self.mean)
        )
        self.mean = self.mean + self.lr * self.velocity

    # -------------- adaptive hook --------------
    def adaptive_pooling(
        self,
        flats: torch.Tensor,
        losses: torch.Tensor,
        accs: Optional[torch.Tensor] = None,
    ) -> None:
        return  # base: do nothing

    # -------------- main update --------------
    def update(self, model_stats_fn: Callable[[nn.Module], SNNStats]):  # type: ignore[override]
        """Evaluate population with *one* forward pass per sample."""
        self._last_stats_fn = model_stats_fn  # cache for _loss_of

        losses: List[torch.Tensor] = []
        accs:   List[torch.Tensor] = []

        for m in self.samples():
            stats = model_stats_fn(m)
            losses.append(stats.loss.detach())
            accs.append(torch.tensor(stats.get_accuracy(), device=self.device))

        losses_t = torch.stack(losses)
        accs_t = torch.stack(accs)

        self._pso_step(losses_t)
        flats = self.mean.unsqueeze(0) + self.std * self._last_eps
        self.adaptive_pooling(flats, losses_t, accs=accs_t)

    # -------------- misc --------------
    def _loss_of(self, flat: torch.Tensor) -> torch.Tensor:
        if self._last_stats_fn is None:
            return torch.tensor(float("inf"), device=self.device)
        stats = self._last_stats_fn(self._flat_to_model(flat))
        return stats.loss.detach()

    def get_best_model(self) -> nn.Module:  # type: ignore[override]
        return self._flat_to_model(self.history_best)


# -----------------------------------------------------------------------------
# PPSOModelWithPooling — adds accuracy‑gated offspring pooling
# -----------------------------------------------------------------------------
class PPSOModelWithPooling(PPSOModel):
    def __init__(self, *args, acc_threshold: float = 0.95, topk_ratio: float = 0.25, **kw):
        super().__init__(*args, **kw)
        self.acc_threshold = acc_threshold
        self.topk_ratio = topk_ratio

    def adaptive_pooling(self, flats: torch.Tensor, losses: torch.Tensor, accs: Optional[torch.Tensor] = None) -> None:  # noqa: N802
        if accs is not None:
            best_acc = accs[torch.argmin(losses)]
            if best_acc >= self.acc_threshold:
                return

        S = flats.size(0)
        k = max(2, int(S * self.topk_ratio))
        top_idx = torch.argsort(losses)[:k]
        topk = flats[top_idx]
        offspring = topk + self.std * torch.randn_like(topk)

        off_losses = torch.stack([self._loss_of(f) for f in offspring])
        best_off = offspring[torch.argmin(off_losses)]

        self.mean = best_off.detach()
        self.velocity.zero_()

        min_off = off_losses.min()
        if min_off < self.history_best_loss:
            self.history_best_loss = min_off
            self.history_best = best_off.clone()
