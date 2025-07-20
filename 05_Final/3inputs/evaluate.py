"""evaluate_incremental.py
================================
Evaluate incremental SOH models and compute ACC, BWT, FWT using MAE.
Supports overriding BASE_DIR via command line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import torch

from model import (
    Config, SOHLSTM, set_seed, Trainer, 
    DataProcessor, Visualizer, create_dataloaders,
)


def compute_gem_scores(error_matrix: np.ndarray) -> dict[str, float]:
    """Compute ACC, BWT, FWT from an error matrix (MAE).
    Rows: 0 = baseline; 1..T = after task0..task_{T-1}
    """
    rows, cols = error_matrix.shape
    assert rows == cols + 1, f"Matrix must be (T+1, T), got {rows}x{cols}"
    T = cols

    acc = float(error_matrix[-1].mean())

    if T > 1:
        # BWT: compare R_{T,j} vs R_{j+1,j} for j=0..T-2
        prev_diag = error_matrix[np.arange(1, T), np.arange(0, T-1)]
        last_row = error_matrix[-1, :T-1]
        bwt = float((last_row - prev_diag).mean())
    else:
        bwt = 0.0

    if T > 1:
        # FWT: compare R_{j,j} vs baseline R_{0,j} for j=1..T-1
        curr_diag = error_matrix[np.arange(1, T), np.arange(1, T)]
        baseline_future = error_matrix[0, 1:]
        fwt = float((curr_diag - baseline_future).mean())
    else:
        fwt = 0.0

    return {"ACC": acc, "BWT": bwt, "FWT": fwt}


def main():
    """Evaluate incremental checkpoints and compute GEM metrics."""
    print("Starting incremental evaluation...")
    set_seed(42)
    config = Config()
    task_ids = ['task0', 'task1', 'task2']
    config.BASE_DIR = Path('model')
    print(f"Using base directory: {config.BASE_DIR}")
    print(f"Task sequence: {task_ids}")

    # Setup directories
    inc_dir = config.BASE_DIR / "fine-tuning" / "incremental"
    inc_dir.mkdir(parents=True, exist_ok=True)
    plots_root = inc_dir / "results"
    plots_root.mkdir(parents=True, exist_ok=True)

    # Prepare data
    print("Preparing data loaders...")
    dp = DataProcessor(config.DATA_DIR, config.RESAMPLE, config)
    datasets = dp.prepare_incremental_data(config.incremental_datasets)
    loaders = create_dataloaders(
        datasets, config.SEQUENCE_LENGTH, config.BATCH_SIZE
    )

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Baseline evaluation (row 0)
    print("Running baseline evaluation...")
    baseline_model = SOHLSTM(
        3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT
    ).to(device)
    baseline_model.eval()
    baseline_trainer = Trainer(baseline_model, device, config, inc_dir)

    baseline_errors = []
    for t in task_ids:
        with torch.no_grad():
            _, _, m = baseline_trainer.evaluate(loaders[f"test{t}"])
        baseline_errors.append(m["MAE"])
        print(f"[Baseline] Task {t} MAE: {m['MAE']:.4e}")

    T = len(task_ids)
    err_matrix = np.zeros((T + 1, T))
    err_matrix[0] = baseline_errors

    metrics_summary = []

    # Evaluate each checkpoint
    for i, task_id in enumerate(task_ids):
        print(f"Evaluating checkpoint for task {task_id}...")
        ckpt = inc_dir / f"task{task_id}_best.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        # Load model
        model = SOHLSTM(
            3, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT
        ).to(device)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state), strict=False)
        model.eval()

        trainer = Trainer(model, device, config, inc_dir)

        # Task-wise eval
        for j, eval_task in enumerate(task_ids):
            with torch.no_grad():
                preds, tgts, m = trainer.evaluate(loaders[f"test_{eval_task}"])
            err_matrix[i + 1, j] = m["MAE"]
            metrics_summary.append({
                "row": i + 1,
                "after_task": task_id,
                "eval_task": eval_task,
                **{"MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"]},
            })
            if eval_task == task_id:
                td = plots_root / f"task{task_id}"
                td.mkdir(exist_ok=True)
                Visualizer.plot_predictions(preds, tgts, m, td)
                Visualizer.plot_prediction_scatter(preds, tgts, td)

        # Full test
        with torch.no_grad():
            _, _, fm = trainer.evaluate(loaders["test_full"])
        metrics_summary.append({
            "row": i + 1,
            "after_task": task_id,
            "eval_task": "full",
            **{"MAE": fm["MAE"], "RMSE": fm["RMSE"], "R2": fm["R2"]},
        })

    # GEM metrics
    gem = compute_gem_scores(err_matrix)
    print(
        "=== GEM (MAE) metrics ===\n"
        f"ACC={gem['ACC']:.4e}  BWT={gem['BWT']:.4e}  FWT={gem['FWT']:.4e}"
    )

    # Save results
    print("Saving result files...")
    np.save(inc_dir / "mae_matrix.npy", err_matrix)
    pd.DataFrame(
        err_matrix, columns=[f"task{t}" for t in task_ids]
    ).to_csv(inc_dir / "mae_matrix.csv", index_label="row")

    (inc_dir / "gem_mae_scores.json").write_text(
        json.dumps(gem, indent=2)
    )

    pd.DataFrame(metrics_summary).to_csv(
        inc_dir / "incremental_metrics_summary.csv", index=False
    )
    print("Incremental evaluation completed.")
    return gem, err_matrix


if __name__ == "__main__":
    main()
