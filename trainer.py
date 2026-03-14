import copy
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    DEVICE, BATCH_SIZE, MAX_EPOCHS,
    USE_FOCAL_LOSS, FOCAL_GAMMA, LABEL_SMOOTHING,
    WEIGHT_DECAY, LR, USE_ONECYCLE, MAX_LR,
    USE_AMP, USE_EMA_WEIGHTS, EMA_DECAY, EARLY_STOPPING_PATIENCE,
    USE_COSINE_ANNEALING, WARMUP_EPOCHS,
    GRAD_ACCUM_STEPS, USE_MIXUP, MIXUP_ALPHA,
)
from utils import FocalLoss, ModelEMA
from model import CNNBiLSTMTransformer
from sequence_dataset import SequenceDataset


def compute_class_weights(y: np.ndarray):
    counts = np.bincount(y, minlength=3).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation for sequence classification."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)  # Ensure lam >= 0.5
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    """Compute loss for mixup."""
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


def train_model(train_ds: SequenceDataset, valid_ds: SequenceDataset, input_size: int):
    """Train model using lazy SequenceDataset.

    Args:
        train_ds: Training SequenceDataset
        valid_ds: Validation SequenceDataset
        input_size: Number of features
    """
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = CNNBiLSTMTransformer(input_size=input_size).to(DEVICE)

    y_train = train_ds.get_all_targets()
    class_weights = compute_class_weights(y_train)

    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ── Scheduler ──
    scheduler = None
    if USE_COSINE_ANNEALING:
        # Cosine annealing with warmup
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS * max(1, len(train_loader)),
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(MAX_EPOCHS - WARMUP_EPOCHS) * max(1, len(train_loader)),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[WARMUP_EPOCHS * max(1, len(train_loader))],
        )
    elif USE_ONECYCLE:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=MAX_LR,
            steps_per_epoch=max(1, len(train_loader)), epochs=MAX_EPOCHS,
        )

    scaler_amp = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA_WEIGHTS else None

    best_val_loss = float("inf")
    best_state = None
    patience_left = EARLY_STOPPING_PATIENCE
    history = []

    total_batches = len(train_loader)
    print(f"Training on {DEVICE.upper()} | {total_batches} batches/epoch | batch_size={BATCH_SIZE}")

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_count = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            # ── Mixup augmentation ──
            if USE_MIXUP:
                xb, yb_a, yb_b, lam = mixup_data(xb, yb, MIXUP_ALPHA)
            else:
                yb_a, yb_b, lam = yb, yb, 1.0

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(xb)
                if USE_MIXUP:
                    loss = mixup_criterion(criterion, logits, yb_a, yb_b, lam)
                else:
                    loss = criterion(logits, yb)
                loss = loss / GRAD_ACCUM_STEPS

            scaler_amp.scale(loss).backward()

            # ── Gradient accumulation ──
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()
            if ema is not None:
                ema.update(model)

            train_loss += loss.item() * GRAD_ACCUM_STEPS * len(xb)
            train_count += len(xb)

            if (step + 1) % 50 == 0 or (step + 1) == total_batches:
                elapsed = time.time() - epoch_start
                avg_loss = train_loss / max(train_count, 1)
                print(f"  epoch {epoch+1} | batch {step+1}/{total_batches} | loss={avg_loss:.5f} | {elapsed:.0f}s", end="\r")

        train_loss /= max(train_count, 1)

        # ── Apply EMA before validation ──
        model.eval()
        if ema is not None:
            ema.apply_to(model)

        valid_loss = 0.0
        valid_count = 0

        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                valid_loss += loss.item() * len(xb)
                valid_count += len(xb)

        valid_loss /= max(valid_count, 1)

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = EARLY_STOPPING_PATIENCE
            improved = True
        else:
            patience_left -= 1
            improved = False

        # Restore original weights for continued training
        if ema is not None:
            ema.restore(model)

        epoch_time = time.time() - epoch_start
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "valid_loss": float(valid_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_sec": float(epoch_time),
            "patience_left": int(patience_left),
            "improved": bool(improved),
        })
        improved_mark = "*" if improved else ""
        print(f"epoch={epoch+1}/{MAX_EPOCHS} train_loss={train_loss:.5f} valid_loss={valid_loss:.5f} time={epoch_time:.0f}s pat={patience_left} {improved_mark}")

        if patience_left <= 0:
            print("Early stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.training_history = history
    model.best_val_loss = float(best_val_loss)
    return model


def predict_proba(model, dataset_or_features, batch_size: int = 1024,
                  targets=None, seq_len=None, start=0, count=None):
    """Predict probabilities. Accepts either a SequenceDataset or raw scaled 2D features.

    Usage 1: predict_proba(model, dataset)
    Usage 2: predict_proba(model, scaled_features_2d, targets=t, seq_len=s, start=0, count=n)
    """
    if isinstance(dataset_or_features, SequenceDataset):
        ds = dataset_or_features
    else:
        # Legacy: accept 2D or 3D array
        arr = dataset_or_features
        if arr.ndim == 3:
            # Legacy 3D input — wrap in a simple dataset
            from torch.utils.data import Dataset as _DS

            class _LegacyDS(_DS):
                def __init__(self, x):
                    self.x = x

                def __len__(self):
                    return len(self.x)

                def __getitem__(self, idx):
                    return torch.from_numpy(self.x[idx].copy()).float(), torch.tensor(0, dtype=torch.long)

            ds = _LegacyDS(arr)
        else:
            if targets is None:
                targets = np.zeros(len(arr), dtype=np.int64)
            if count is None:
                count = len(arr) - (seq_len or 1) + 1
            ds = SequenceDataset(arr, targets, seq_len, start, count)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    model.eval()
    probs_all = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())

    return np.concatenate(probs_all, axis=0)


def evaluate(model, dataset_or_features, y=None, **kwargs):
    """Evaluate model. Accepts SequenceDataset or legacy arrays."""
    if isinstance(dataset_or_features, SequenceDataset):
        probs = predict_proba(model, dataset_or_features)
        y_true = dataset_or_features.get_all_targets()
    else:
        probs = predict_proba(model, dataset_or_features, **kwargs)
        y_true = y

    y_pred = probs.argmax(axis=1)
    return {
        "report_text": classification_report(y_true, y_pred, digits=4),
        "report": classification_report(y_true, y_pred, digits=4, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
