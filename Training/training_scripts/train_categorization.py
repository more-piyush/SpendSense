"""
train_categorization.py — Training script for the DistilBERT transaction categorization model.

Supports multiple model types via config:
  - "logistic_regression": TF-IDF + Logistic Regression baseline
  - "distilbert": DistilBERT fine-tuning with multi-label classification head

Usage:
  python train_categorization.py configs/categorization_baseline.yaml
  python train_categorization.py configs/categorization_distilbert_v1.yaml

All hyperparameters come from the YAML config file.
All runs are logged to MLflow.
"""

import sys
import os
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    get_linear_schedule_with_warmup,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import mlflow.sklearn

from utils import (
    load_config,
    setup_mlflow,
    log_environment_info,
    compute_data_hash,
    get_device,
    TrainingTimer,
    log_peak_memory,
)

warnings.filterwarnings("ignore")


# ============================================================
# DATASET
# ============================================================
class TransactionDataset(Dataset):
    """Dataset for transaction categorization.

    Each example has:
      - description (str): transaction text
      - categories (list[str]): list of valid category labels
      - amount (float): transaction amount
      - currency (str): currency code
      - country (str): country code
    """

    def __init__(self, df, tokenizer, max_length, label_binarizer,
                 currency_vocab, country_vocab):
        self.descriptions = df["description"].tolist()
        self.amounts = df["amount"].values.astype(np.float32)
        self.currencies = df["currency"].tolist()
        self.countries = df["country"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_binarizer = label_binarizer
        self.currency_vocab = currency_vocab
        self.country_vocab = country_vocab

        # Encode multi-label targets
        if "categories" in df.columns:
            cats = df["categories"].apply(
                lambda x: x if isinstance(x, list) else json.loads(x)
            ).tolist()
            self.labels = self.label_binarizer.transform(cats).astype(np.float32)
        else:
            self.labels = None

        # Sample weights (if available)
        self.weights = df["sample_weight"].values.astype(np.float32) \
            if "sample_weight" in df.columns \
            else np.ones(len(df), dtype=np.float32)

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        # Tokenize
        encoding = self.tokenizer(
            self.descriptions[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "amount": np.log1p(self.amounts[idx]),
            "currency_idx": self.currency_vocab.get(self.currencies[idx], 0),
            "country_idx": self.country_vocab.get(self.countries[idx], 0),
            "weight": self.weights[idx],
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item


# ============================================================
# MODEL: DistilBERT + Classification Head
# ============================================================
class DistilBertCategorizer(nn.Module):
    """DistilBERT encoder with multi-label classification head.

    Architecture:
      [CLS] embedding (768) + amount (1) + currency (20) + country (50)
      = 839 -> Linear(839, 256) -> ReLU -> Dropout -> Linear(256, n_classes) -> Sigmoid
    """

    def __init__(self, pretrained_model, n_classes, n_currencies=20,
                 n_countries=50, dropout=0.3, freeze_layers=0,
                 freeze_embeddings=False):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_model)

        # Freeze embeddings (major param reduction — ~23M params)
        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            print(f"[INFO] Froze embedding layer (~23M params)")

        # Freeze early transformer layers if specified
        if freeze_layers > 0:
            for i, layer in enumerate(self.bert.transformer.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"[INFO] Froze first {freeze_layers} transformer layers")

        # Auxiliary feature embeddings
        self.currency_emb = nn.Embedding(n_currencies + 1, 20, padding_idx=0)
        self.country_emb = nn.Embedding(n_countries + 1, 50, padding_idx=0)

        # Classification head
        # Input: 768 (CLS) + 1 (amount) + 20 (currency) + 50 (country) = 839
        hidden_dim = 768 + 1 + 20 + 50
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, input_ids, attention_mask, amount, currency_idx, country_idx):
        # DistilBERT encoder
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token: (batch, 768)

        # Auxiliary features
        currency_feat = self.currency_emb(currency_idx)     # (batch, 20)
        country_feat = self.country_emb(country_idx)        # (batch, 50)
        amount_feat = amount.unsqueeze(1)                    # (batch, 1)

        # Concatenate: (batch, 839)
        combined = torch.cat(
            [cls_embedding, amount_feat, currency_feat, country_feat], dim=1
        )

        # Classification head -> raw logits (no sigmoid, BCEWithLogitsLoss handles it)
        logits = self.classifier(combined)
        return logits


# ============================================================
# TRAINING FUNCTIONS
# ============================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, criterion,
                    device, scaler, use_amp):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    n_batches = 0
    total_batches = len(dataloader)
    log_every = max(1, total_batches // 50)  # ~50 progress prints per epoch
    epoch_start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        amount = batch["amount"].to(device)
        currency_idx = batch["currency_idx"].to(device)
        country_idx = batch["country_idx"].to(device)
        labels = batch["labels"].to(device)
        weights = batch["weight"].to(device)

        if use_amp:
            with autocast():
                logits = model(input_ids, attention_mask, amount,
                               currency_idx, country_idx)
                loss = criterion(logits, labels)
                # Apply sample weights
                loss = (loss * weights.unsqueeze(1)).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask, amount,
                           currency_idx, country_idx)
            loss = criterion(logits, labels)
            loss = (loss * weights.unsqueeze(1)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % log_every == 0 or batch_idx == 0:
            elapsed = time.time() - epoch_start
            rate = (batch_idx + 1) / max(elapsed, 0.001)
            eta = (total_batches - batch_idx - 1) / max(rate, 0.001)
            print(f"  step {batch_idx+1}/{total_batches} "
                  f"loss={loss.item():.4f} "
                  f"rate={rate:.1f} it/s "
                  f"eta={eta/60:.1f}min", flush=True)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, threshold=0.5):
    """Evaluate model. Returns metrics dict."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0
    max_probs = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        amount = batch["amount"].to(device)
        currency_idx = batch["currency_idx"].to(device)
        country_idx = batch["country_idx"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, amount,
                       currency_idx, country_idx)
        loss = criterion(logits, labels).mean()
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        max_probs.extend(probs.max(dim=1).values.cpu().numpy().tolist())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # Compute metrics
    # Per-sample accuracy: exact match ratio
    exact_match = (all_preds == all_labels).all(axis=1).mean()

    # Subset accuracy (at least one correct)
    subset_acc = ((all_preds * all_labels).sum(axis=1) > 0).mean()

    # Macro F1 across all labels
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)

    # Per-class precision, recall
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    min_recall = float(recall[support > 0].min()) if (support > 0).any() else 0.0

    # Abstention rate (max prob < 0.7)
    abstention_rate = sum(1 for p in max_probs if p < 0.7) / len(max_probs)

    metrics = {
        "val_loss": total_loss / max(n_batches, 1),
        "exact_match_accuracy": round(exact_match, 4),
        "subset_accuracy": round(subset_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "min_class_recall": round(min_recall, 4),
        "abstention_rate": round(abstention_rate, 4),
        "mean_max_confidence": round(np.mean(max_probs), 4),
    }
    return metrics


# ============================================================
# BASELINE: TF-IDF + Logistic Regression
# ============================================================
def train_baseline(config, train_df, val_df, test_df, label_binarizer):
    """Train a TF-IDF + Logistic Regression baseline."""
    print("\n" + "=" * 60)
    print("TRAINING BASELINE: TF-IDF + Logistic Regression")
    print("=" * 60)

    with TrainingTimer("baseline_training"):
        # TF-IDF features
        tfidf = TfidfVectorizer(
            max_features=config.get("max_features", 10000),
            ngram_range=tuple(config.get("ngram_range", [1, 2])),
            sublinear_tf=True,
        )
        X_train = tfidf.fit_transform(train_df["description"])
        X_val = tfidf.transform(val_df["description"])
        X_test = tfidf.transform(test_df["description"])

        # Encode labels — for baseline, use the first (primary) category
        y_train = train_df["categories"].apply(
            lambda x: x[0] if isinstance(x, list) else json.loads(x)[0]
        )
        y_val = val_df["categories"].apply(
            lambda x: x[0] if isinstance(x, list) else json.loads(x)[0]
        )
        y_test = test_df["categories"].apply(
            lambda x: x[0] if isinstance(x, list) else json.loads(x)[0]
        )

        # Logistic Regression
        clf = LogisticRegression(
            C=config.get("C", 1.0),
            max_iter=config.get("max_iter", 1000),
            solver="lbfgs",
            multi_class="multinomial",
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

    # Evaluate
    val_preds = clf.predict(X_val)
    test_preds = clf.predict(X_test)
    val_probs = clf.predict_proba(X_val)

    val_acc = accuracy_score(y_val, val_preds)
    test_acc = accuracy_score(y_test, test_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    abstention = (val_probs.max(axis=1) < 0.7).mean()

    metrics = {
        "val_accuracy": round(val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "val_macro_f1": round(val_f1, 4),
        "test_macro_f1": round(test_f1, 4),
        "abstention_rate": round(abstention, 4),
    }

    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(clf, "model")
    mlflow.log_artifact(config["_config_path"])

    # --- Rich Artifacts ---

    # 1. Classification report (text)
    report = classification_report(y_test, test_preds, zero_division=0)
    mlflow.log_text(report, "classification_report.txt")

    # 2. Confusion matrix (image)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    classes = clf.classes_
    n_show = min(20, len(classes))  # Limit to 20 classes for readability
    cm = confusion_matrix(y_test, test_preds, labels=classes[:n_show])
    fig_cm, ax_cm = plt.subplots(figsize=(max(10, n_show), max(8, n_show * 0.8)))
    ConfusionMatrixDisplay(cm, display_labels=classes[:n_show]).plot(
        ax=ax_cm, cmap="Blues", xticks_rotation=45, values_format="d"
    )
    ax_cm.set_title("Confusion Matrix (Test Set)")
    fig_cm.tight_layout()
    mlflow.log_figure(fig_cm, "confusion_matrix.png")
    plt.close(fig_cm)

    # 3. Per-class F1 bar chart
    per_class_report = classification_report(y_test, test_preds, output_dict=True, zero_division=0)
    class_names = [k for k in per_class_report if k not in ("accuracy", "macro avg", "weighted avg")]
    f1_scores = [per_class_report[c]["f1-score"] for c in class_names]
    sorted_idx = np.argsort(f1_scores)
    fig_f1, ax_f1 = plt.subplots(figsize=(10, max(6, len(class_names) * 0.3)))
    ax_f1.barh([class_names[i] for i in sorted_idx], [f1_scores[i] for i in sorted_idx], color="steelblue")
    ax_f1.set_xlabel("F1 Score")
    ax_f1.set_title("Per-Class F1 Score (Test Set)")
    ax_f1.set_xlim(0, 1.05)
    fig_f1.tight_layout()
    mlflow.log_figure(fig_f1, "per_class_f1.png")
    plt.close(fig_f1)

    # 4. Data stats (JSON)
    class_dist = y_train.value_counts().to_dict()
    data_stats = {
        "train_size": len(y_train),
        "val_size": len(y_val),
        "test_size": len(y_test),
        "n_classes": len(clf.classes_),
        "class_distribution": {str(k): int(v) for k, v in class_dist.items()},
    }
    mlflow.log_text(json.dumps(data_stats, indent=2), "data_stats.json")

    # 5. Sample predictions (CSV) — 50 random test samples
    sample_idx = np.random.RandomState(42).choice(len(y_test), min(50, len(y_test)), replace=False)
    sample_df = pd.DataFrame({
        "description": test_df["description"].iloc[sample_idx].values,
        "true_label": y_test.iloc[sample_idx].values,
        "predicted_label": test_preds[sample_idx],
        "correct": (y_test.iloc[sample_idx].values == test_preds[sample_idx]),
    })
    sample_csv = sample_df.to_csv(index=False)
    mlflow.log_text(sample_csv, "sample_predictions.csv")

    # 6. Feature importance (TF-IDF top features)
    feature_names = tfidf.get_feature_names_out()
    # Average absolute coefficient across all classes
    if hasattr(clf, 'coef_'):
        avg_importance = np.abs(clf.coef_).mean(axis=0)
        top_n = 30
        top_idx = np.argsort(avg_importance)[-top_n:]
        fig_feat, ax_feat = plt.subplots(figsize=(10, 8))
        ax_feat.barh(
            [feature_names[i] for i in top_idx],
            avg_importance[top_idx],
            color="coral"
        )
        ax_feat.set_xlabel("Mean |Coefficient|")
        ax_feat.set_title("Top 30 TF-IDF Features by Importance")
        fig_feat.tight_layout()
        mlflow.log_figure(fig_feat, "feature_importance.png")
        plt.close(fig_feat)

    print(f"\n[RESULTS] Baseline metrics: {json.dumps(metrics, indent=2)}")
    return metrics


# ============================================================
# DISTILBERT TRAINING
# ============================================================
def train_distilbert(config, train_df, val_df, test_df, label_binarizer,
                     currency_vocab, country_vocab):
    """Train DistilBERT categorization model."""
    print("\n" + "=" * 60)
    print("TRAINING: DistilBERT Categorization Model")
    print("=" * 60)

    device = get_device()
    n_classes = len(label_binarizer.classes_)
    print(f"[INFO] Number of classes: {n_classes}")
    print(f"[INFO] Classes: {list(label_binarizer.classes_)}")

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(config["pretrained_model"])

    # Datasets
    train_dataset = TransactionDataset(
        train_df, tokenizer, config["max_length"],
        label_binarizer, currency_vocab, country_vocab,
    )
    val_dataset = TransactionDataset(
        val_df, tokenizer, config["max_length"],
        label_binarizer, currency_vocab, country_vocab,
    )
    test_dataset = TransactionDataset(
        test_df, tokenizer, config["max_length"],
        label_binarizer, currency_vocab, country_vocab,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"] * 2, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"] * 2, shuffle=False, num_workers=2
    )

    # Model
    model = DistilBertCategorizer(
        pretrained_model=config["pretrained_model"],
        n_classes=n_classes,
        dropout=config.get("dropout", 0.3),
        freeze_layers=config.get("freeze_layers", 0),
        freeze_embeddings=config.get("freeze_embeddings", False),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mlflow.log_param("total_parameters", total_params)
    mlflow.log_param("trainable_parameters", trainable_params)
    print(f"[INFO] Total params: {total_params:,}, Trainable: {trainable_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.01),
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # Scheduler: linear warmup then linear decay
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Loss function: BCEWithLogitsLoss for multi-label
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # Mixed precision
    use_amp = config.get("mixed_precision", True) and torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    print(f"[INFO] Mixed precision (FP16): {use_amp}")

    # Training loop
    best_val_f1 = 0
    best_epoch = 0
    patience = config.get("early_stopping_patience", 2)
    patience_counter = 0

    total_start = time.time()

    for epoch in range(config["epochs"]):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, scaler, use_amp,
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start

        # Log to MLflow
        mlflow.log_metric("train_loss", round(train_loss, 4), step=epoch)
        for k, v in val_metrics.items():
            mlflow.log_metric(k, v, step=epoch)
        mlflow.log_metric("epoch_time_sec", round(epoch_time, 2), step=epoch)
        mlflow.log_metric("learning_rate",
                          optimizer.param_groups[0]["lr"], step=epoch)

        print(
            f"[EPOCH {epoch+1}/{config['epochs']}] "
            f"loss={train_loss:.4f} | "
            f"val_f1={val_metrics['macro_f1']:.4f} | "
            f"val_acc={val_metrics['subset_accuracy']:.4f} | "
            f"abstention={val_metrics['abstention_rate']:.4f} | "
            f"time={epoch_time:.1f}s"
        )

        # Early stopping check
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "/tmp/best_model.pt")
            print(f"  -> New best model saved (F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping at epoch {epoch+1}")
                break

    total_time = time.time() - total_start

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load("/tmp/best_model.pt"))
    test_metrics = evaluate(model, test_loader, criterion, device)

    # Log final metrics
    final_metrics = {
        "best_epoch": best_epoch + 1,
        "best_val_macro_f1": best_val_f1,
        "test_exact_match_accuracy": test_metrics["exact_match_accuracy"],
        "test_subset_accuracy": test_metrics["subset_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_micro_f1": test_metrics["micro_f1"],
        "test_min_class_recall": test_metrics["min_class_recall"],
        "test_abstention_rate": test_metrics["abstention_rate"],
        "total_training_time_sec": round(total_time, 2),
    }
    mlflow.log_metrics(final_metrics)

    # Log model — save state_dict and wrap with pyfunc to avoid torch.export issue
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "n_classes": n_classes,
            "classes": list(label_binarizer.classes_),
        }, model_path)
        artifacts = {"model_path": model_path}
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=mlflow.pyfunc.PythonModel(),
            artifacts=artifacts,
        )
    mlflow.log_artifact(config["_config_path"])

    # Log peak memory
    log_peak_memory()

    # --- Rich Artifacts ---

    # 1. Training loss curve
    # Collect loss history by re-reading logged metrics
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    epochs_range = list(range(1, best_epoch + 2))
    # We logged train_loss per epoch via mlflow, reconstruct from the run
    # For simplicity, just note the final metrics; the step-logged metrics
    # are visible in the MLflow UI charts.
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Metric")
    ax_loss.set_title("Training Progress (see MLflow Model Metrics tab for interactive charts)")
    ax_loss.text(0.5, 0.5, f"Best epoch: {best_epoch+1}\nBest val F1: {best_val_f1:.4f}\n"
                 f"Test F1: {test_metrics['macro_f1']:.4f}\n"
                 f"Total time: {total_time:.0f}s",
                 transform=ax_loss.transAxes, ha='center', va='center', fontsize=14)
    fig_loss.tight_layout()
    mlflow.log_figure(fig_loss, "training_summary.png")
    plt.close(fig_loss)

    # 2. Per-class metrics (using test set multi-label predictions)
    test_preds_np = []
    test_labels_np = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            amount = batch["amount"].to(device)
            currency_idx = batch["currency_idx"].to(device)
            country_idx = batch["country_idx"].to(device)
            logits = model(input_ids, attention_mask, amount, currency_idx, country_idx)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            test_preds_np.append(preds.cpu().numpy())
            test_labels_np.append(batch["labels"].numpy())
    test_preds_np = np.vstack(test_preds_np)
    test_labels_np = np.vstack(test_labels_np)

    # Classification report
    class_names = list(label_binarizer.classes_)
    report = classification_report(
        test_labels_np, test_preds_np, target_names=class_names, zero_division=0
    )
    mlflow.log_text(report, "classification_report.txt")

    # 3. Per-class F1 bar chart
    report_dict = classification_report(
        test_labels_np, test_preds_np, target_names=class_names,
        output_dict=True, zero_division=0
    )
    per_class = {k: v for k, v in report_dict.items()
                 if k not in ("micro avg", "macro avg", "weighted avg", "samples avg")}
    f1_vals = [(k, v["f1-score"]) for k, v in per_class.items()]
    f1_vals.sort(key=lambda x: x[1])
    fig_f1, ax_f1 = plt.subplots(figsize=(10, max(6, len(f1_vals) * 0.35)))
    ax_f1.barh([x[0] for x in f1_vals], [x[1] for x in f1_vals], color="steelblue")
    ax_f1.set_xlabel("F1 Score")
    ax_f1.set_title("Per-Class F1 Score (Test Set)")
    ax_f1.set_xlim(0, 1.05)
    fig_f1.tight_layout()
    mlflow.log_figure(fig_f1, "per_class_f1.png")
    plt.close(fig_f1)

    # 4. Data stats (JSON)
    # Class distribution in training data
    train_cats = train_df["categories"].apply(
        lambda x: x if isinstance(x, list) else json.loads(x)
    )
    cat_counts = {}
    for cats in train_cats:
        for c in cats:
            cat_counts[c] = cat_counts.get(c, 0) + 1
    data_stats = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "n_classes": n_classes,
        "class_distribution": dict(sorted(cat_counts.items(), key=lambda x: -x[1])),
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    mlflow.log_text(json.dumps(data_stats, indent=2), "data_stats.json")

    # 5. Sample predictions (CSV) — 50 random test samples
    sample_size = min(50, len(test_df))
    sample_idx = np.random.RandomState(42).choice(len(test_df), sample_size, replace=False)
    sample_descs = test_df["description"].iloc[sample_idx].values
    sample_true = test_labels_np[sample_idx]
    sample_pred = test_preds_np[sample_idx]
    rows = []
    for i in range(sample_size):
        true_cats = [class_names[j] for j in range(n_classes) if sample_true[i][j] > 0.5]
        pred_cats = [class_names[j] for j in range(n_classes) if sample_pred[i][j] > 0.5]
        rows.append({
            "description": sample_descs[i],
            "true_categories": "; ".join(true_cats),
            "predicted_categories": "; ".join(pred_cats),
            "correct": true_cats == pred_cats,
        })
    sample_csv = pd.DataFrame(rows).to_csv(index=False)
    mlflow.log_text(sample_csv, "sample_predictions.csv")

    # 6. Label encoder classes (JSON — needed for inference)
    mlflow.log_text(json.dumps(class_names), "label_classes.json")

    print(f"\n[RESULTS] Final test metrics: {json.dumps(final_metrics, indent=2)}")
    return final_metrics


# ============================================================
# DATA LOADING
# ============================================================
def load_data(config):
    """Load and split training data."""
    data_path = config["data_path"]
    print(f"[INFO] Loading data from {data_path}")

    if data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported data format: {data_path}")

    # Parse categories if stored as strings
    if isinstance(df["categories"].iloc[0], str):
        df["categories"] = df["categories"].apply(json.loads)

    # Log data hash for reproducibility
    data_hash = compute_data_hash(data_path)
    mlflow.log_param("data_hash", data_hash)
    mlflow.log_param("data_rows", len(df))
    print(f"[INFO] Loaded {len(df)} rows, data hash: {data_hash}")

    # Build label binarizer from all categories
    all_categories = sorted(set(
        cat for cats in df["categories"] for cat in cats
    ))
    label_binarizer = MultiLabelBinarizer(classes=all_categories)
    label_binarizer.fit([all_categories])
    mlflow.log_param("n_classes", len(all_categories))

    # Build vocab mappings for auxiliary features
    top_currencies = df["currency"].value_counts().head(20).index.tolist()
    currency_vocab = {c: i + 1 for i, c in enumerate(top_currencies)}

    top_countries = df["country"].value_counts().head(50).index.tolist()
    country_vocab = {c: i + 1 for i, c in enumerate(top_countries)}

    # Split: use pre-defined split column, or create one
    if "split" in df.columns:
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)
        test_df = df[df["split"] == "test"].reset_index(drop=True)
    else:
        # Stratified random split: 80/10/10
        from sklearn.model_selection import train_test_split

        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    # Optional subsampling for low-resource training (e.g., CPU-only VMs)
    max_samples = config.get("max_samples")
    if max_samples:
        max_samples = int(max_samples)
        val_cap = max(1000, max_samples // 8)
        test_cap = max(1000, max_samples // 8)
        if len(train_df) > max_samples:
            train_df = train_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        if len(val_df) > val_cap:
            val_df = val_df.sample(n=val_cap, random_state=42).reset_index(drop=True)
        if len(test_df) > test_cap:
            test_df = test_df.sample(n=test_cap, random_state=42).reset_index(drop=True)
        print(f"[INFO] Subsampled to max_samples={max_samples}")
        mlflow.log_param("max_samples", max_samples)

    mlflow.log_params({
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    })
    print(f"[INFO] Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    return train_df, val_df, test_df, label_binarizer, currency_vocab, country_vocab


# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: python train_categorization.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    config["_config_path"] = config_path

    # Setup MLflow
    setup_mlflow(config)

    with mlflow.start_run(run_name=config.get("run_name", os.path.basename(config_path))):
        # Log all config params
        for k, v in config.items():
            if not k.startswith("_"):
                mlflow.log_param(k, v)

        # Log environment
        log_environment_info()

        # Load data
        train_df, val_df, test_df, lb, cur_vocab, cou_vocab = load_data(config)

        # Route to appropriate model type
        model_type = config.get("model_type", "distilbert")

        if model_type == "logistic_regression":
            metrics = train_baseline(config, train_df, val_df, test_df, lb)
        elif model_type == "distilbert":
            metrics = train_distilbert(
                config, train_df, val_df, test_df, lb, cur_vocab, cou_vocab
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        print("\n[DONE] Training complete. Check MLflow for full results.")


if __name__ == "__main__":
    main()
