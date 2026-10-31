import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train Experiment 5 from scratch")
    parser.add_argument("--train-dir", type=Path, default=Path("../train"),
                        help="ImageFolder directory with 16 class subfolders")
    parser.add_argument("--output-dir", type=Path, default=Path("training_outputs"),
                        help="Separate output directory; existing best_model.pth is never overwritten")
    args = parser.parse_args()
    if not args.train_dir.is_dir():
        parser.error(f"Training directory not found: {args.train_dir}")

    import os, time, random, copy, math, json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from PIL import Image, ImageFilter
    from torch.utils.data import Dataset, DataLoader, Subset
    from torchvision import transforms, models
    from torchvision.datasets import ImageFolder
    from timm.data import Mixup
    from sklearn.model_selection import train_test_split
    from timm.loss import SoftTargetCrossEntropy

    # ============================================================
    # CONFIGURATION
    # ============================================================

    E5_SEED = 42
    random.seed(E5_SEED)
    np.random.seed(E5_SEED)
    torch.manual_seed(E5_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(E5_SEED)

    e5_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    torch.backends.cudnn.benchmark = True

    E5_SIZE = 224
    E5_SSL_BATCH = 64
    E5_FT_BATCH = 32

    E5_SSL_EPOCHS = 100
    E5_FT_EPOCHS = 50
    E5_FT_PATIENCE = 8

    E5_QUEUE_SIZE = 4096
    E5_FEATURE_DIM = 128
    E5_MOMENTUM = 0.999
    E5_TEMPERATURE = 0.2

    E5_DIR = str(args.output_dir)
    os.makedirs(E5_DIR, exist_ok=True)

    E5_SSL_BEST = f"{E5_DIR}/moco_v2_best.pth"
    E5_SSL_LAST = f"{E5_DIR}/moco_v2_latest.pth"
    E5_FT_BEST = f"{E5_DIR}/resnet18_moco_finetuned_best.pth"
    E5_FT_LAST = f"{E5_DIR}/finetune_latest.pth"

    E5_MEAN = [0.485, 0.456, 0.406]
    E5_STD = [0.229, 0.224, 0.225]

    train_dir = str(args.train_dir)
    base_dataset = ImageFolder(train_dir)
    labels = np.array(base_dataset.targets)
    indices = np.arange(len(base_dataset))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.20, stratify=labels, random_state=42
    )

    # ============================================================
    # AUGMENTATION
    # ============================================================

    class E5GaussianBlur:
        def __init__(self, radius=(0.1, 2.0)):
            self.radius = radius

        def __call__(self, image):
            sigma = random.uniform(*self.radius)
            return image.filter(
                ImageFilter.GaussianBlur(radius=sigma)
            )

    # Two independently augmented views for MoCo
    e5_ssl_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            E5_SIZE,
            scale=(0.4, 1.0)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            )
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            E5GaussianBlur()
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(E5_MEAN, E5_STD)
    ])

    # Supervised fine-tuning augmentation
    e5_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            E5_SIZE,
            scale=(0.75, 1.0),
            ratio=(0.85, 1.15)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.12,
                hue=0.02
            )
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(E5_MEAN, E5_STD),
        transforms.RandomErasing(
            p=0.10,
            scale=(0.02, 0.08)
        )
    ])

    e5_eval_transform = transforms.Compose([
        transforms.Resize((E5_SIZE, E5_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(E5_MEAN, E5_STD)
    ])

    # ============================================================
    # DATASETS
    # ============================================================

    e5_train_base = ImageFolder(train_dir)
    e5_classes = e5_train_base.classes
    E5_NUM_CLASSES = len(e5_classes)

    assert E5_NUM_CLASSES == 16

    e5_train_idx = list(map(int, train_idx))
    e5_val_idx = list(map(int, val_idx))

    assert not set(e5_train_idx).intersection(e5_val_idx)
    assert len(e5_train_idx) == 1920
    assert len(e5_val_idx) == 480

    class E5TwoViews(Dataset):
        def __init__(self, base, indices, transform):
            self.base = base
            self.indices = indices
            self.transform = transform

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):
            actual_index = self.indices[index]
            path, _ = self.base.samples[actual_index]

            with Image.open(path) as image:
                image = image.convert("RGB")
                q = self.transform(image)
                k = self.transform(image)

            return q, k

    e5_ssl_dataset = E5TwoViews(
        e5_train_base,
        e5_train_idx,
        e5_ssl_transform
    )

    e5_supervised_train = ImageFolder(
        train_dir,
        transform=e5_train_transform
    )

    e5_supervised_val = ImageFolder(
        train_dir,
        transform=e5_eval_transform
    )

    e5_ssl_loader = DataLoader(
        e5_ssl_dataset,
        batch_size=E5_SSL_BATCH,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(e5_device.type == "cuda")
    )

    e5_train_loader = DataLoader(
        Subset(e5_supervised_train, e5_train_idx),
        batch_size=E5_FT_BATCH,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(e5_device.type == "cuda")
    )

    e5_val_loader = DataLoader(
        Subset(e5_supervised_val, e5_val_idx),
        batch_size=E5_FT_BATCH,
        shuffle=False,
        num_workers=0,
        pin_memory=(e5_device.type == "cuda")
    )

    # ============================================================
    # RESNET-18 ENCODER
    # ============================================================

    def e5_make_encoder():
        encoder = models.resnet18(weights=None)

        feature_dim = encoder.fc.in_features

        # MoCo v2 two-layer projection head
        encoder.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, E5_FEATURE_DIM)
        )

        return encoder

    # ============================================================
    # MoCo v2 MODEL
    # ============================================================

    class E5MoCo(nn.Module):

        def __init__(self):
            super().__init__()

            self.encoder_q = e5_make_encoder()
            self.encoder_k = e5_make_encoder()

            # Initialize key encoder from query encoder
            for q, k in zip(
                self.encoder_q.parameters(),
                self.encoder_k.parameters()
            ):
                k.data.copy_(q.data)
                k.requires_grad = False

            queue = F.normalize(
                torch.randn(
                    E5_FEATURE_DIM,
                    E5_QUEUE_SIZE
                ),
                dim=0
            )

            self.register_buffer("queue", queue)
            self.register_buffer(
                "queue_ptr",
                torch.zeros(1, dtype=torch.long)
            )

        @torch.no_grad()
        def momentum_update(self):

            for q, k in zip(
                self.encoder_q.parameters(),
                self.encoder_k.parameters()
            ):
                k.data.mul_(E5_MOMENTUM).add_(
                    q.data,
                    alpha=1.0 - E5_MOMENTUM
                )

            # Copy running BatchNorm statistics
            for q_buffer, k_buffer in zip(
                self.encoder_q.buffers(),
                self.encoder_k.buffers()
            ):
                k_buffer.copy_(q_buffer)

        @torch.no_grad()
        def enqueue(self, keys):

            keys = keys.detach()
            batch_size = keys.shape[0]

            ptr = int(self.queue_ptr.item())
            end = ptr + batch_size

            if end <= E5_QUEUE_SIZE:
                self.queue[:, ptr:end] = keys.T
            else:
                first = E5_QUEUE_SIZE - ptr
                self.queue[:, ptr:] = keys[:first].T
                self.queue[:, :end - E5_QUEUE_SIZE] = keys[first:].T

            self.queue_ptr[0] = end % E5_QUEUE_SIZE

        def forward(self, im_q, im_k):

            q = F.normalize(
                self.encoder_q(im_q),
                dim=1
            )

            with torch.no_grad():
                self.momentum_update()

                # Shuffle keys before BatchNorm, then undo shuffle
                permutation = torch.randperm(
                    im_k.size(0),
                    device=im_k.device
                )
                inverse = torch.argsort(permutation)

                k = self.encoder_k(im_k[permutation])
                k = F.normalize(k, dim=1)
                k = k[inverse]

            positive = torch.sum(
                q * k,
                dim=1,
                keepdim=True
            )

            negative = torch.mm(
                q,
                self.queue.clone().detach()
            )

            logits = torch.cat(
                [positive, negative],
                dim=1
            ) / E5_TEMPERATURE

            labels = torch.zeros(
                logits.size(0),
                dtype=torch.long,
                device=logits.device
            )

            loss = F.cross_entropy(logits, labels)

            self.enqueue(k)

            return loss

    e5_moco = E5MoCo().to(e5_device)

    print("Experiment 5 initialized successfully.")
    print("Device:", e5_device)
    print("Architecture: ResNet-18")
    print("SSL: MoCo v2")
    print("Training images:", len(e5_ssl_dataset))
    print("Validation images:", len(e5_val_idx))
    print("Classes:", E5_NUM_CLASSES)
    print("MoCo queue:", E5_QUEUE_SIZE)
    print("SSL epochs:", E5_SSL_EPOCHS)
    print("Fine-tuning epochs:", E5_FT_EPOCHS)

    # ============================================================
    # EXPERIMENT 5 - MoCo v2 SELF-SUPERVISED PRETRAINING
    # ============================================================

    e5_ssl_optimizer = torch.optim.SGD(
        e5_moco.encoder_q.parameters(),
        lr=0.03,
        momentum=0.9,
        weight_decay=1e-4
    )

    e5_ssl_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        e5_ssl_optimizer,
        T_max=E5_SSL_EPOCHS,
        eta_min=1e-5
    )

    e5_ssl_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(e5_device.type == "cuda")
    )

    e5_ssl_history = []
    e5_ssl_start_epoch = 0
    e5_ssl_best_loss = float("inf")

    # Resume if an earlier SSL run was interrupted
    if os.path.exists(E5_SSL_LAST):

        state = torch.load(
            E5_SSL_LAST,
            map_location="cpu",
            weights_only=False
        )

        e5_moco.load_state_dict(state["model"])
        e5_ssl_optimizer.load_state_dict(state["optimizer"])
        e5_ssl_scheduler.load_state_dict(state["scheduler"])

        if state.get("scaler") is not None:
            e5_ssl_scaler.load_state_dict(state["scaler"])

        e5_ssl_start_epoch = state["epoch"]
        e5_ssl_best_loss = state["best_loss"]
        e5_ssl_history = state["history"]

        print(
            "Resuming MoCo pretraining from epoch",
            e5_ssl_start_epoch
        )

    e5_ssl_start_time = time.time()

    for epoch in range(
        e5_ssl_start_epoch,
        E5_SSL_EPOCHS
    ):

        e5_moco.train()

        # Key encoder is used only for forward inference
        e5_moco.encoder_k.eval()

        total_loss = 0.0
        total_images = 0

        for images_q, images_k in e5_ssl_loader:

            images_q = images_q.to(
                e5_device,
                non_blocking=True
            )
            images_k = images_k.to(
                e5_device,
                non_blocking=True
            )

            e5_ssl_optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=e5_device.type,
                enabled=(e5_device.type == "cuda")
            ):
                loss = e5_moco(images_q, images_k)

            e5_ssl_scaler.scale(loss).backward()
            e5_ssl_scaler.unscale_(e5_ssl_optimizer)

            torch.nn.utils.clip_grad_norm_(
                e5_moco.encoder_q.parameters(),
                max_norm=5.0
            )

            e5_ssl_scaler.step(e5_ssl_optimizer)
            e5_ssl_scaler.update()

            total_loss += loss.item() * images_q.size(0)
            total_images += images_q.size(0)

        avg_loss = total_loss / total_images

        e5_ssl_scheduler.step()

        e5_ssl_history.append({
            "epoch": epoch + 1,
            "ssl_loss": avg_loss
        })

        # Save latest state for resuming
        torch.save({
            "epoch": epoch + 1,
            "model": e5_moco.state_dict(),
            "optimizer": e5_ssl_optimizer.state_dict(),
            "scheduler": e5_ssl_scheduler.state_dict(),
            "scaler": e5_ssl_scaler.state_dict(),
            "best_loss": min(e5_ssl_best_loss, avg_loss),
            "history": e5_ssl_history
        }, E5_SSL_LAST)

        if avg_loss < e5_ssl_best_loss:

            e5_ssl_best_loss = avg_loss

            torch.save(
                e5_moco.encoder_q.state_dict(),
                E5_SSL_BEST
            )

        pd.DataFrame(e5_ssl_history).to_csv(
            f"{E5_DIR}/moco_v2_history.csv",
            index=False
        )

        print(
            f"SSL Epoch {epoch+1:03d}/{E5_SSL_EPOCHS} | "
            f"Loss: {avg_loss:.4f} | "
            f"Best Loss: {e5_ssl_best_loss:.4f}"
        )

    e5_ssl_minutes = (
        time.time() - e5_ssl_start_time
    ) / 60

    print("\nMoCo v2 pretraining completed.")
    print("Best SSL training loss:", round(e5_ssl_best_loss, 4))
    print("Time:", round(e5_ssl_minutes, 2), "minutes")
    print("Saved encoder:", E5_SSL_BEST)

    # ============================================================
    # EXPERIMENT 5 - ADVANCED SUPERVISED FINE-TUNING
    # ============================================================

    assert os.path.exists(E5_SSL_BEST), (
        "MoCo encoder checkpoint missing. "
        "Complete Section 9.2 first."
    )

    # ------------------------------------------------------------
    # 1. BUILD CLASSIFIER
    # ------------------------------------------------------------

    e5_classifier = models.resnet18(
        weights=None
    )

    e5_classifier.fc = nn.Linear(
        e5_classifier.fc.in_features,
        E5_NUM_CLASSES
    )

    # Load backbone weights from MoCo query encoder
    e5_ssl_weights = torch.load(
        E5_SSL_BEST,
        map_location="cpu",
        weights_only=True
    )

    backbone_weights = {
        key: value
        for key, value in e5_ssl_weights.items()
        if not key.startswith("fc.")
    }

    load_result = e5_classifier.load_state_dict(
        backbone_weights,
        strict=False
    )

    assert set(load_result.missing_keys) == {
        "fc.weight", "fc.bias"
    }, load_result.missing_keys

    assert not load_result.unexpected_keys

    e5_classifier = e5_classifier.to(e5_device)

    # ------------------------------------------------------------
    # 2. MIXUP AND CUTMIX
    # ------------------------------------------------------------

    e5_mixup = Mixup(
        mixup_alpha=0.2,
        cutmix_alpha=0.5,
        prob=0.65,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=0.05,
        num_classes=E5_NUM_CLASSES
    )

    e5_train_criterion = SoftTargetCrossEntropy()
    e5_val_criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------
    # 3. DISCRIMINATIVE LEARNING RATES
    # ------------------------------------------------------------

    e5_ft_optimizer = torch.optim.AdamW([
        {
            "params": [
                p for name, p in e5_classifier.named_parameters()
                if not name.startswith("fc.")
            ],
            "lr": 3e-4
        },
        {
            "params": e5_classifier.fc.parameters(),
            "lr": 1e-3
        }
    ], weight_decay=0.02)

    e5_ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        e5_ft_optimizer,
        T_max=E5_FT_EPOCHS,
        eta_min=1e-6
    )

    e5_ft_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(e5_device.type == "cuda")
    )

    # ------------------------------------------------------------
    # 4. RESUME SUPPORT
    # ------------------------------------------------------------

    e5_ft_history = []
    e5_ft_start_epoch = 0
    e5_ft_best_val = -1.0
    e5_ft_best_epoch = 0
    e5_ft_no_improve = 0

    if os.path.exists(E5_FT_LAST):

        state = torch.load(
            E5_FT_LAST,
            map_location="cpu",
            weights_only=False
        )

        e5_classifier.load_state_dict(state["model"])
        e5_ft_optimizer.load_state_dict(state["optimizer"])
        e5_ft_scheduler.load_state_dict(state["scheduler"])

        if state.get("scaler") is not None:
            e5_ft_scaler.load_state_dict(state["scaler"])

        e5_ft_start_epoch = state["epoch"]
        e5_ft_best_val = state["best_val"]
        e5_ft_best_epoch = state["best_epoch"]
        e5_ft_no_improve = state["no_improve"]
        e5_ft_history = state["history"]

        print(
            "Resuming fine-tuning from epoch",
            e5_ft_start_epoch
        )

    # ------------------------------------------------------------
    # 5. TRAINING
    # ------------------------------------------------------------

    e5_ft_start_time = time.time()

    for epoch in range(
        e5_ft_start_epoch,
        E5_FT_EPOCHS
    ):

        e5_classifier.train()

        running_train_loss = 0.0
        train_total = 0

        for images, labels in e5_train_loader:

            images = images.to(
                e5_device,
                non_blocking=True
            )
            labels = labels.to(
                e5_device,
                non_blocking=True
            )

            images, soft_labels = e5_mixup(
                images, labels
            )

            e5_ft_optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=e5_device.type,
                enabled=(e5_device.type == "cuda")
            ):
                logits = e5_classifier(images)

                loss = e5_train_criterion(
                    logits,
                    soft_labels
                )

            e5_ft_scaler.scale(loss).backward()
            e5_ft_scaler.unscale_(e5_ft_optimizer)

            torch.nn.utils.clip_grad_norm_(
                e5_classifier.parameters(),
                max_norm=1.0
            )

            e5_ft_scaler.step(e5_ft_optimizer)
            e5_ft_scaler.update()

            running_train_loss += (
                loss.item() * images.size(0)
            )

            train_total += images.size(0)

        train_loss = running_train_loss / train_total

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        e5_classifier.eval()

        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in e5_val_loader:

                images = images.to(e5_device)
                labels = labels.to(e5_device)

                with torch.autocast(
                    device_type=e5_device.type,
                    enabled=(e5_device.type == "cuda")
                ):
                    logits = e5_classifier(images)
                    loss = e5_val_criterion(
                        logits, labels
                    )

                val_loss_sum += (
                    loss.item() * labels.size(0)
                )

                val_correct += (
                    logits.argmax(1) == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = 100 * val_correct / val_total

        e5_ft_history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        # --------------------------------------------------------
        # BEST CHECKPOINT
        # --------------------------------------------------------

        if val_acc > e5_ft_best_val:

            e5_ft_best_val = val_acc
            e5_ft_best_epoch = epoch + 1
            e5_ft_no_improve = 0

            torch.save(
                e5_classifier.state_dict(),
                E5_FT_BEST
            )

        else:
            e5_ft_no_improve += 1

        e5_ft_scheduler.step()

        # Save resumable checkpoint
        torch.save({
            "epoch": epoch + 1,
            "model": e5_classifier.state_dict(),
            "optimizer": e5_ft_optimizer.state_dict(),
            "scheduler": e5_ft_scheduler.state_dict(),
            "scaler": e5_ft_scaler.state_dict(),
            "best_val": e5_ft_best_val,
            "best_epoch": e5_ft_best_epoch,
            "no_improve": e5_ft_no_improve,
            "history": e5_ft_history
        }, E5_FT_LAST)

        pd.DataFrame(e5_ft_history).to_csv(
            f"{E5_DIR}/finetune_history.csv",
            index=False
        )

        print(
            f"Epoch {epoch+1:02d}/{E5_FT_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Best: {e5_ft_best_val:.2f}%"
        )

        if e5_ft_no_improve >= E5_FT_PATIENCE:
            print("Early stopping triggered.")
            break

    e5_ft_minutes = (
        time.time() - e5_ft_start_time
    ) / 60

    print("\nFine-tuning completed.")
    print("Best epoch:", e5_ft_best_epoch)
    print(f"Best validation accuracy: {e5_ft_best_val:.2f}%")
    print(f"Fine-tuning time: {e5_ft_minutes:.2f} minutes")
    print("Best checkpoint:", E5_FT_BEST)

if __name__ == "__main__":
    main()