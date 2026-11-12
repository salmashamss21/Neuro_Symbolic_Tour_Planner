"""Training and evaluation pipeline for the Two-Head Neural Network vs. Baselines."""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

from data.loader import TourDataset
from nn.baselines import MarkovCountBaseline, PopularityBaseline
from nn.model import DualHeadTourNN, encode_features


def evaluate_predictions(
    probs: np.ndarray,
    target_next: np.ndarray,
    pred_log_dur: np.ndarray,
    target_log_dur: np.ndarray,
) -> dict[str, float]:
    """Computes NLL, Top-1 Accuracy, Top-3 Accuracy, Log-Duration RMSE, and Minutes MAE."""
    bs = len(target_next)
    nll = -float(np.mean(np.log(np.clip(probs[np.arange(bs), target_next], 1e-12, 1.0))))
    top1 = float(np.mean(np.argmax(probs, axis=1) == target_next))
    top3_idx = np.argsort(probs, axis=1)[:, -3:]
    top3 = float(np.mean([target_next[i] in top3_idx[i] for i in range(bs)]))

    log_rmse = float(math.sqrt(np.mean((pred_log_dur - target_log_dur) ** 2)))
    pred_min = np.exp(pred_log_dur) / 60.0
    true_min = np.exp(target_log_dur) / 60.0
    mae_min = float(np.mean(np.abs(pred_min - true_min)))
    rmse_min = float(math.sqrt(np.mean((pred_min - true_min) ** 2)))
    return {
        "nll": round(nll, 4),
        "top1_acc": round(top1 * 100.0, 2),
        "top3_acc": round(top3 * 100.0, 2),
        "log_dur_rmse": round(log_rmse, 4),
        "dur_mae_min": round(mae_min, 2),
        "dur_rmse_min": round(rmse_min, 2),
    }


def train_and_evaluate(
    dataset: TourDataset,
    seed: int = 42,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 0.004,
    save_dir: str | Path = "outputs/models",
) -> tuple[DualHeadTourNN, MarkovCountBaseline, PopularityBaseline, dict]:
    """Trains DualHeadTourNN and benchmarks it against Markov and Popularity baselines."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cat_to_idx = {c: i for i, c in enumerate(dataset.categories)}
    n_pois = len(dataset.poi_ids)
    n_cats = len(dataset.categories)

    def prepare_split(split_name: str):
        df = dataset.trans_df[dataset.trans_df["split"] == split_name]
        curr_idx = df["currentPoiID"].map(dataset.poi_to_idx).to_numpy(dtype=np.int64)
        cat_idx = df["currentCategory"].map(cat_to_idx).to_numpy(dtype=np.int64)
        hours = df["hour"].to_numpy(dtype=np.int64)
        next_idx = df["nextPoiID"].map(dataset.poi_to_idx).to_numpy(dtype=np.int64)
        log_dur = df["logDuration"].to_numpy(dtype=np.float64)
        x = encode_features(curr_idx, cat_idx, hours, n_pois=n_pois, n_cats=n_cats)
        return x, curr_idx, cat_idx, hours, next_idx, log_dur

    x_tr, c_tr, cat_tr, _, y_next_tr, y_dur_tr = prepare_split("train")
    x_va, c_va, cat_va, _, y_next_va, y_dur_va = prepare_split("val")
    x_te, c_te, cat_te, _, y_next_te, y_dur_te = prepare_split("test")

    # Fit baselines on training split
    markov = MarkovCountBaseline(n_pois=n_pois, smoothing=1.0)
    markov.fit(c_tr, y_next_tr, y_dur_tr)

    pop_base = PopularityBaseline(n_pois=n_pois, n_cats=n_cats, smoothing=1.0)
    pop_base.fit(c_tr, y_next_tr, cat_tr, y_dur_tr)

    # Train DualHeadTourNN
    model = DualHeadTourNN(n_pois=n_pois, n_cats=n_cats, hidden_dim=64, seed=seed)
    rng = np.random.default_rng(seed)

    history = {"train_nll": [], "val_nll": [], "train_mse": [], "val_mse": []}
    best_val_loss = float("inf")
    best_weights_path = save_dir / f"dual_head_nn_seed{seed}.npz"

    n_train = len(x_tr)
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n_train)
        ep_nlls, ep_mses = [], []
        cur_lr = lr * (0.985 ** (epoch - 1))
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            nll, mse = model.train_step(
                x_tr[idx], c_tr[idx], y_next_tr[idx], y_dur_tr[idx], lr=cur_lr, weight_decay=1e-4
            )
            ep_nlls.append(nll)
            ep_mses.append(mse)

        probs_va, _ = model.forward_next_place(x_va, c_va)
        pred_dur_va, _ = model.forward_duration(x_va)
        val_metrics = evaluate_predictions(probs_va, y_next_va, pred_dur_va, y_dur_va)
        val_mse = float(np.mean((pred_dur_va - y_dur_va) ** 2))

        history["train_nll"].append(round(float(np.mean(ep_nlls)), 4))
        history["val_nll"].append(val_metrics["nll"])
        history["train_mse"].append(round(float(np.mean(ep_mses)), 4))
        history["val_mse"].append(round(val_mse, 4))

        combo_loss = val_metrics["nll"] + val_mse
        if combo_loss < best_val_loss:
            best_val_loss = combo_loss
            model.save(best_weights_path)

    model.load(best_weights_path)

    # Evaluate on held-out Test split
    probs_te_nn, _ = model.forward_next_place(x_te, c_te)
    dur_te_nn, _ = model.forward_duration(x_te)
    nn_test_metrics = evaluate_predictions(probs_te_nn, y_next_te, dur_te_nn, y_dur_te)

    probs_te_mk = markov.predict_next_probs_batch(c_te)
    dur_te_mk = markov.predict_log_duration_batch(c_te)
    mk_test_metrics = evaluate_predictions(probs_te_mk, y_next_te, dur_te_mk, y_dur_te)

    probs_te_pop = pop_base.predict_next_probs_batch(c_te)
    dur_te_pop = pop_base.predict_log_duration_batch(cat_te)
    pop_test_metrics = evaluate_predictions(probs_te_pop, y_next_te, dur_te_pop, y_dur_te)

    results = {
        "seed": seed,
        "test_samples": int(len(x_te)),
        "nn_test": nn_test_metrics,
        "markov_baseline_test": mk_test_metrics,
        "popularity_baseline_test": pop_test_metrics,
        "history": history,
    }
    with open(save_dir / f"nn_eval_seed{seed}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return model, markov, pop_base, results


if __name__ == "__main__":
    ds = TourDataset("data")
    _, _, _, res = train_and_evaluate(ds, seed=42)
    print("NN Test Metrics:", res["nn_test"])
    print("Markov Baseline Test Metrics:", res["markov_baseline_test"])
    print("Popularity Baseline Test Metrics:", res["popularity_baseline_test"])
