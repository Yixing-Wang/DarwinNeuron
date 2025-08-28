from time import time
import torch as T
import numpy as np
from torch.optim import Adam
from src.LandscapeAnalysis.Pipeline import NNProblemConfig 
from src.Utilities import run_snn_on_batch, evaluate_snn
from torch.nn.utils import parameters_to_vector

def train_with_sgd(problem_id: int, batch_size: int, algorithm_params: dict, term_tuple: tuple,
                   seed: int, batch_logger, log_dir, db_path: str = 'data/LandscapeAnalysis.db'):

    # ----- setup -----
    problem_config = NNProblemConfig.lookup_by_id(problem_id, db_path)
    loader = problem_config.get_loader(batch_size, db_path)

    term_type, term_value = term_tuple
    if term_type != 'n_gen':
        raise ValueError(f"Unsupported termination type {term_type}. Expected 'n_gen'.")
    total_epochs = int(term_value)
    total_batches = total_epochs * len(loader)
    device = "cuda" if T.cuda.is_available() else "cpu"

    model = problem_config.get_model(db_path).to(device)
    loss_fn = problem_config.get_loss(db_path)

    if algorithm_params['optimizer'] == 'adam':
        optimizer = Adam(model.parameters(), lr=algorithm_params['lr'])
    else:
        raise ValueError(f"Unsupported optimizer {algorithm_params['optimizer']}.")

    # ----- training loop -----
    model.train()
    t0 = time()

    gen_counter = 0  
    for epoch in range(total_epochs):
        for bidx, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True) if hasattr(x, "to") else x
            y = y.to(device, non_blocking=True) if hasattr(y, "to") else y
            # forward/backward
            stats = run_snn_on_batch(model, x, y, loss_fn)
            stats.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            is_epoch_end = (bidx == len(loader) - 1)

            # epoch end
            epoch_acc_full = None
            if is_epoch_end:
                ep_stats = evaluate_snn(model, loader, loss_fn, device)
                epoch_acc_full = ep_stats.get_accuracy()

            acc_this  = stats.get_accuracy()
            loss_this = float(stats.loss.item())

            batch_logger.write(
                gen=gen_counter,
                n_evals=gen_counter,      
                epoch=epoch,
                batch=bidx,   
                best_f=loss_this,
                best_acc=acc_this,
                avg_acc=acc_this,  
                wall_time=time() - t0,
                best_x=parameters_to_vector(model.parameters()).detach().cpu().numpy(),
                is_epoch_end=is_epoch_end,
                epoch_acc_full=epoch_acc_full
            )

            gen_counter += 1

    # ----- finish -----
    x_best = parameters_to_vector(model.parameters()).detach().cpu().numpy()
    x_best_file = str(log_dir / "x_best_final.npy")
    np.save(x_best_file, x_best.astype(np.float32))

    return {
        "best_f": float(loss_this),
        "best_acc": float(acc_this),
        "final_epoch_acc_full": float(epoch_acc_full),
        "n_evals": gen_counter - 1,
        "n_gen": gen_counter - 1,
        "total_epoch": total_epochs,
        "x_best_len": int(x_best.size),
        "x_best_file": x_best_file,
        "runtime": time() - t0
    }


# from time import time
# from dataclasses import dataclass
# from torch.optim import Adam
# from src.OptimizerAnalysis.callback_and_runner import GenLogger
# from src.LandscapeAnalysis.Pipeline import NNProblemConfig 
# from src.Utilities import run_snn_on_batch, evaluate_snn
# from torch.nn.utils import parameters_to_vector


# def train_with_sgd(problem_id: int, batch_size: int, algorithm_params: dict, term_tuple: tuple, seed: int, batch_logger: GenLogger, log_dir: str, db_path: str = 'data/LandscapeAnalysis.db'):
#     ## set up data, model, and loss function
#     problem_config = NNProblemConfig.lookup_by_id(problem_id, db_path)
#     loader = problem_config.get_loader(batch_size, db_path)
#     term_tuple[1] *= len(loader)
#     model = problem_config.get_model(db_path)
#     loss = problem_config.get_loss(db_path)
#     if algorithm_params['optimizer'] == 'adam':
#         optimizer = Adam(model.parameters(), lr=algorithm_params['lr'])
#     else:
#         raise ValueError(f"Unsupported optimizer {algorithm_params['optimizer']}.")

#     # set up total epochs
#     if term_tuple[0] != 'n_gen':
#         raise ValueError(f"Unsupported termination type {term_tuple[0]}. Expected 'n_gen'.")
#     total_batches = term_tuple[1]
#     total_epochs = total_batches // len(loader)

#     ## run the optimization
#     model.train()
#     t_init = time()
#     for epoch in range(total_epochs):
#         for batch, (x, y) in enumerate(loader):
#             # Forward pass
#             stats = run_snn_on_batch(model, x, y, loss)

#             # Backward pass and optimization
#             stats.loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()

#             # Log the training information
#             batch_logger.write(
#                 gen=epoch*len(loader) + batch, 
#                 n_evals=epoch*len(loader) + batch,
#                 best_f=stats.loss,
#                 best_acc=stats.get_accuracy(),
#                 avg_acc=stats.get_accuracy(),
#                 wall_time = time() - t_init,
#                 epoch = epoch,
#                 batch = epoch*len(loader)+batch,
#                 best_x = parameters_to_vector(model.parameters()).detach().cpu().numpy(),
#                 is_epoch_end = (batch == len(loader)-1),
#                 epoch_acc_full=None
#             )
#         # calculate and log epoch accuracy
#         epoch_stats = evaluate_snn(model, loader, loss)
#         epoch_acc_full = epoch_stats.get_accuracy()
        

#     x_best = parameters_to_vector(model.parameters()).detach().cpu().numpy()
#     x_best_file = str(log_dir / "x_best_final.npy")
#     np.save(x_best_file, x_best.astype(np.float32))
#     # return the training result
#     return {
#         "best_f": stats.loss.item(),
#         "best_acc": stats.get_accuracy(),
#         "n_evals": epoch * len(loader) + batch,
#         "n_gen": epoch * len(loader) + batch,
#         "total_epoch": total_epochs,
#         "x_best_len": len(x_best),
#         "X_best_file": x_best_file,
#         "runtime": time() - t_init
#     }