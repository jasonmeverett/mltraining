import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from data.medical import HeartFailurePredictionDataset
from models.classification import HeartDiseaseClassifier


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"


# Create the dataset
data = HeartFailurePredictionDataset()
train_loader = data.train_dataloader()
val_loader = data.val_dataloader()

# Create model!
model = HeartDiseaseClassifier(
    emb_dim=16,
    hidden_dim=256,
    dropout=0.1
)
model.to(device)

# Create optimizer and loss fn
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()

writer = SummaryWriter(log_dir="runs/01_heart_disease_norm_e16_hd256_d01_lr001")

max_epochs = 100
global_step = 0

for epoch in range(max_epochs):
    # Training
    model.train()
    running_train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch[-1].reshape(-1, 1).to(device)
        X = data.prepare_batch(batch[:-1], device)
        logits = model(*X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        global_step += 1
        running_train_loss += loss.item() * y.size(0)
        writer.add_scalar("Loss/train_step", loss.item(), global_step)
    avg_train_loss = running_train_loss / len(train_loader.dataset)
    writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)

    # Validation
    model.eval()

    running_val_loss = 0.0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            y = batch[-1].reshape(-1, 1).to(device)
            X = data.prepare_batch(batch[:-1], device)
            logits = model(*X)
            probs = torch.sigmoid(logits).view(-1)
            loss = loss_fn(logits, y)
            running_val_loss += loss.item() * y.size(0)
            labels = y.view(-1)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    avg_val_loss = running_val_loss / len(val_loader.dataset)

    all_probs = torch.cat(all_probs).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_preds = (all_probs >= 0.5).astype("int32")

    # Metrics
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, zero_division=0)
    val_recall = recall_score(all_labels, all_preds, zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        val_roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        val_roc_auc = float("nan")
    try:
        val_pr_auc = average_precision_score(all_labels, all_probs)
    except ValueError:
        val_pr_auc = float("nan")

    # Log to TensorBoard
    writer.add_scalar("Metrics/loss", avg_val_loss, epoch)
    writer.add_scalar("Metrics/accuracy", val_accuracy, epoch)
    writer.add_scalar("Metrics/precision", val_precision, epoch)
    writer.add_scalar("Metrics/recall", val_recall, epoch)
    writer.add_scalar("Metrics/f1", val_f1, epoch)
    writer.add_scalar("Metrics/roc_auc", val_roc_auc, epoch)
    writer.add_scalar("Metrics/pr_auc", val_pr_auc, epoch)

    print(
        f"Epoch [{epoch+1}/{max_epochs}] "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Acc: {val_accuracy:.4f} | "
        f"Prec: {val_precision:.4f} | "
        f"Rec: {val_recall:.4f} | "
        f"F1: {val_f1:.4f} | "
        f"ROC-AUC: {val_roc_auc:.4f} | "
        f"PR-AUC: {val_pr_auc:.4f}"
    )
