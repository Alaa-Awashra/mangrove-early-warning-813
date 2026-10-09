from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


def _prepare_fold(frame: pd.DataFrame, columns: list[str], train: np.ndarray,
                  evaluation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    training = scaler.fit_transform(imputer.fit_transform(frame.iloc[train][columns]))
    held_out = scaler.transform(imputer.transform(frame.iloc[evaluation][columns]))
    return training.astype("float32"), held_out.astype("float32")


def run_cnn(frame: pd.DataFrame, target: np.ndarray, spatial: dict, plan_sha256: str,
            output_dir: Path, seeds: tuple[int, ...] = (813, 814, 815)) -> pd.DataFrame:
    import torch
    from torch import nn

    torch.set_num_threads(2)
    output_dir.mkdir(parents=True, exist_ok=True)
    hsi_columns = [column for column in frame if column.startswith("h1_")]
    baseline_columns = [column for column in frame if column.startswith(("f1_", "f2_", "f3_"))]
    if not hsi_columns or not baseline_columns:
        raise ValueError("CNN requires HSI and conventional feature families")

    class Network(nn.Module):
        def __init__(self, with_baseline: bool):
            super().__init__()
            self.with_baseline = with_baseline
            self.spectral = nn.Sequential(nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
                                          nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
                                          nn.AdaptiveAvgPool1d(1), nn.Flatten())
            if with_baseline:
                self.context = nn.Sequential(nn.Linear(len(baseline_columns), 16), nn.ReLU())
            self.head = nn.Sequential(nn.Linear(48 if with_baseline else 32, 16), nn.ReLU(), nn.Linear(16, 1))

        def forward(self, spectrum, context):
            spectral = self.spectral(spectrum[:, None, :])
            if self.with_baseline:
                spectral = torch.cat((spectral, self.context(context)), dim=1)
            return self.head(spectral).squeeze(1)

    def fit_predict(train_spectrum, train_context, train_target, validation_spectrum,
                    validation_context, validation_target, test_spectrum, test_context,
                    with_baseline, seed, epochs, use_early_stopping):
        torch.manual_seed(seed)
        network = Network(with_baseline)
        positive = max(1, int(train_target.sum()))
        negative = max(1, len(train_target) - positive)
        loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, dtype=torch.float32))
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=0.001)
        generator = torch.Generator().manual_seed(seed)
        train_spectrum = torch.as_tensor(train_spectrum)
        train_context = torch.as_tensor(train_context)
        train_target = torch.as_tensor(train_target.astype("float32"))
        validation_spectrum = torch.as_tensor(validation_spectrum)
        validation_context = torch.as_tensor(validation_context)
        test_spectrum = torch.as_tensor(test_spectrum)
        test_context = torch.as_tensor(test_context)
        best_loss = float("inf")
        best_epoch = 1
        best_state = None
        stale = 0
        log = []
        for epoch in range(1, epochs + 1):
            network.train()
            order = torch.randperm(len(train_target), generator=generator)
            batch_losses = []
            for batch in order.split(256):
                optimizer.zero_grad()
                loss = loss_function(network(train_spectrum[batch], train_context[batch]), train_target[batch])
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach()))
            network.eval()
            with torch.no_grad():
                validation_loss = float(loss_function(
                    network(validation_spectrum, validation_context),
                    torch.as_tensor(validation_target.astype("float32"))))
            log.append((epoch, float(np.mean(batch_losses)), validation_loss))
            if validation_loss < best_loss - 1e-4:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {key: value.clone() for key, value in network.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if use_early_stopping and stale >= 3:
                break
        if use_early_stopping:
            network.load_state_dict(best_state)
        network.eval()
        with torch.no_grad():
            probability = torch.sigmoid(network(test_spectrum, test_context)).numpy()
        return probability, best_epoch, log

    predictions = []
    training_logs = []
    for with_baseline in (False, True):
        model_name = "cnn_plus_baseline" if with_baseline else "cnn_hsi_only"
        for seed in seeds:
            for fold, (outer_train, outer_test) in enumerate(spatial["splits"]):
                inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed + fold)
                possible = list(inner.split(outer_train, target[outer_train], spatial["groups"][outer_train]))
                inner_train, inner_validation = next(
                    ((outer_train[train], outer_train[validation]) for train, validation in possible
                     if target[outer_train[train]].sum() > 0 and target[outer_train[validation]].sum() > 0),
                    (None, None))
                if inner_train is None:
                    raise ValueError("CNN inner spatial validation lacks positive cases")
                spectrum_train, spectrum_validation = _prepare_fold(frame, hsi_columns, inner_train, inner_validation)
                _, spectrum_test = _prepare_fold(frame, hsi_columns, inner_train, outer_test)
                context_train, context_validation = _prepare_fold(frame, baseline_columns, inner_train, inner_validation)
                _, context_test = _prepare_fold(frame, baseline_columns, inner_train, outer_test)
                _, best_epoch, history = fit_predict(
                    spectrum_train, context_train, target[inner_train], spectrum_validation, context_validation,
                    target[inner_validation], spectrum_test, context_test, with_baseline,
                    seed + fold, 15, True)
                for epoch, training_loss, validation_loss in history:
                    training_logs.append({"model": model_name, "seed": seed, "fold": fold, "epoch": epoch,
                                          "training_loss": training_loss, "validation_loss": validation_loss,
                                          "analysis_plan_sha256": plan_sha256})
                full_spectrum, test_spectrum = _prepare_fold(frame, hsi_columns, outer_train, outer_test)
                full_context, test_context = _prepare_fold(frame, baseline_columns, outer_train, outer_test)
                probability, _, _ = fit_predict(
                    full_spectrum, full_context, target[outer_train],
                    spectrum_validation, context_validation, target[inner_validation],
                    test_spectrum, test_context, with_baseline, seed + fold,
                    best_epoch, False)
                predictions.extend({"sample_index": int(index), "fold": fold, "seed": seed,
                                    "model": model_name, "risk": float(score), "target": int(target[index]),
                                    "analysis_plan_sha256": plan_sha256}
                                   for index, score in zip(outer_test, probability))
                print(f"CNN complete: {model_name} seed {seed} fold {fold}", flush=True)
    pd.DataFrame(training_logs).to_csv(output_dir / "training_logs.csv", index=False)
    return pd.DataFrame(predictions)
