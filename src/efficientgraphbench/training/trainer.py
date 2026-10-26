import logging
import time

import torch
import torch.nn.functional as F

from efficientgraphbench.profiling.detail import epoch as execution_epoch  # noqa: E402
from efficientgraphbench.profiling.detail import operation
from efficientgraphbench.profiling.events import observed, span
from efficientgraphbench.profiling.runtime import synchronize
from efficientgraphbench.profiling.timing import timed
from efficientgraphbench.training.evaluator import accuracy, classification_metrics

logger = logging.getLogger(__name__)


def _train(model, data, config, device, checkpoint, timings):
    optimizer = getattr(torch.optim, config.optimizer)(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
        if config.scheduler == "cosine"
        else None
    )
    best_score, best_epoch, best_state = -1.0, 0, None
    epoch_times, validation_times = [], []
    stale_epochs, termination = 0, "maximum_epochs"
    synchronize(device)
    total_start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        with execution_epoch(epoch, device):
            with timed(timings, "optimization_seconds", device):
                model.train()
                synchronize(device)
                start = time.perf_counter()
                with operation("zero_gradients"):
                    optimizer.zero_grad(set_to_none=True)
                with operation("training_forward"):
                    logits = model(data)
                with operation("loss_calculation"):
                    loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
                if not bool(torch.isfinite(loss)):
                    raise ValueError("training loss is not finite")
                with operation("training_backward"):
                    loss.backward()
                with operation("optimizer_step"):
                    optimizer.step()
                synchronize(device)
            epoch_times.append(time.perf_counter() - start)
            with timed(timings, "validation_seconds", device):
                model.eval()
                start = time.perf_counter()
                with torch.inference_mode():
                    score = accuracy(model(data), data.y, data.val_mask)
                synchronize(device)
            validation_times.append(time.perf_counter() - start)
            if score > best_score:
                stale_epochs = 0
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale_epochs += 1
            if scheduler is not None:
                with operation("scheduler_step"):
                    scheduler.step()
            if epoch == 1 or epoch % 25 == 0 or epoch == config.epochs:
                logger.info("epoch=%d loss=%.4f val_accuracy=%.4f", epoch, loss.item(), score)
            if (
                config.early_stopping_patience is not None
                and stale_epochs >= config.early_stopping_patience
            ):
                termination = "early_stopping"
                break
    synchronize(device)
    total_time = time.perf_counter() - total_start
    checkpoint_start = time.perf_counter()
    with span("checkpoint_save"):
        torch.save(
            {
                "model_state_dict": best_state,
                "best_epoch": best_epoch,
                "validation_accuracy": best_score,
            },
            checkpoint,
        )
    checkpoint_time = time.perf_counter() - checkpoint_start
    test_start = time.perf_counter()
    with span(
        "checkpoint_restore_and_test",
        synchronization="existing device completion boundary (CUDA sync only on CUDA)",
    ):
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            test_metrics = classification_metrics(model(data), data.y, data.test_mask)
        synchronize(device)
    test_time = time.perf_counter() - test_start
    return {
        "checkpoint_write_time_sec": checkpoint_time,
        "checkpoint_restore_and_test_time_sec": test_time,
        "gradient_memory_mib": sum(
            p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None
        )
        / 1024**2,
        "optimizer_tensor_memory_mib": sum(
            v.numel() * v.element_size()
            for state in optimizer.state.values()
            for v in state.values()
            if torch.is_tensor(v)
        )
        / 1024**2,
        "training_time_sec": total_time,
        "epochs_run": epoch,
        "maximum_epochs": config.epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "training_termination_reason": termination,
        "optimizer": config.optimizer,
        "learning_rate": config.lr,
        "weight_decay": config.weight_decay,
        "scheduler": config.scheduler,
        **test_metrics,
        "validation_accuracy": best_score,
        "best_epoch": best_epoch,
        "train_time_sec": total_time,
        "optimization_time_sec": sum(epoch_times),
        "optimization_seconds": timings["optimization_seconds"],
        "validation_seconds": timings["validation_seconds"],
        "total_training_phase_seconds": total_time,
        "checkpoint_save_seconds": checkpoint_time,
        "post_adaptation_evaluation_seconds": test_time,
        "epoch_time_sec": sum(epoch_times) / len(epoch_times),
        "validation_time_sec": sum(validation_times),
        "epoch_times_sec": epoch_times,
    }


@observed("training")
def train(model, data, config, device, checkpoint):
    timings = {}
    started = time.perf_counter()
    try:
        return _train(model, data, config, device, checkpoint, timings)
    except BaseException as exc:
        exc.progressive_measurements = {
            **timings,
            "failed_attempt_elapsed_seconds": time.perf_counter() - started,
        }
        raise
