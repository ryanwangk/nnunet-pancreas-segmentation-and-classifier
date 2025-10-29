import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import wandb
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from .classification_head import ClassificationHead


class MultiTaskTrainer(nnUNetTrainer):
    """
    nnU-Net trainer extended for joint segmentation + classification.
    Tracks segmentation Dice and classification performance in wandb.
    Handles class imbalance via weighted cross-entropy and dropout regularization.
    """

    def __init__(self, plans, configuration_name, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration_name, fold, dataset_json, device)
        self.classification_head = None
        self.num_classes = 3
        self._wandb_init_done = False

        # class weights
        csv_path = os.environ.get("PCCLS_LABELS")
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            counts = df["Subtype"].value_counts().sort_index()
            weights = 1.0 / counts
            weights = weights / weights.sum()
            self.class_weights = torch.tensor(weights.values, dtype=torch.float32).to(device)
            print("Loaded class weights:", self.class_weights)
        else:
            self.class_weights = None
            print("No subtype_labels.csv found — using unweighted loss")

    # initialize
    def build_network(self):
        super().build_network()

        # attach classification head
        last_stage = list(self.network.encoder.stages.values())[-1]
        in_ch = last_stage[-1].conv_block[0].out_channels
        self.classification_head = ClassificationHead(
            in_channels=in_ch,
            num_classes=self.num_classes,
            dropout_p=0.5,
        ).to(self.device)

        print(f"Classification head initialized (in={in_ch}, num_classes={self.num_classes}, dropout=0.5)")

        # wandb
        if not self._wandb_init_done:
            wandb.init(
                project="PancreasSegmentation",
                name=f"Fold{self.fold}_ResEncM",
                config={
                    "trainer": "MultiTaskTrainer",
                    "fold": self.fold,
                    "dropout": 0.5,
                    "loss_weight_cls": 0.1
                },
                reinit=True
            )
            self._wandb_init_done = True

    # compute loss
    def compute_loss(self, data):
        seg_loss = super().compute_loss(data)
        cls_loss, acc = 0.0, None

        if "subtype" in data:
            enc_feat = self.network.encoder_output
            pooled = torch.mean(enc_feat, dim=[2, 3, 4])
            logits = self.classification_head(pooled)
            targets = data["subtype"].long().to(self.device)

            # weighted cross-entropy
            if self.class_weights is not None:
                cls_loss = F.cross_entropy(logits, targets, weight=self.class_weights)
            else:
                cls_loss = F.cross_entropy(logits, targets)

            preds = logits.argmax(dim=1)
            acc = (preds == targets).float().mean().item()

        total_loss = seg_loss + 0.1 * cls_loss if cls_loss != 0.0 else seg_loss #0.1 for cls_loss?
        return total_loss, seg_loss.item(), cls_loss if cls_loss != 0.0 else None, acc

    # training iteration
    def run_training_iteration(self, data):
        self.optimizer.zero_grad(set_to_none=True)
        total_loss, seg_loss, cls_loss, acc = self.compute_loss(data)
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()

        # log to wandb
        log_data = {"train/total_loss": total_loss.item(), "train/seg_loss": seg_loss}
        if cls_loss is not None:
            log_data["train/cls_loss"] = cls_loss.item()
        if acc is not None:
            log_data["train/cls_acc"] = acc
        wandb.log(log_data)

        self.print_to_log_file(
            f"Train total={total_loss.item():.4f}, seg={seg_loss:.4f}, cls={cls_loss if cls_loss else 0:.4f}"
        )
        return total_loss.item()

    # validation loop
    def validate(self):
        self.network.eval()
        val_seg_loss, val_cls_loss, val_acc, count = 0, 0, 0, 0
        total_dice_fg = 0

        with torch.no_grad():
            for data in self.dataloader_val:
                total_loss, seg_loss, cls_loss, acc = self.compute_loss(data)
                val_seg_loss += seg_loss
                if cls_loss is not None:
                    val_cls_loss += cls_loss.item()
                if acc is not None:
                    val_acc += acc

                # segmentation dice
                output = self.network(data["data"])
                pred = torch.argmax(output, dim=1)
                target = data["target"].squeeze(1)
                intersection = torch.sum((pred > 0) & (target > 0)).float()
                union = torch.sum(pred > 0).float() + torch.sum(target > 0).float()
                dice_fg = (2 * intersection / (union + 1e-8)).item()
                total_dice_fg += dice_fg
                count += 1

        val_seg_loss /= count
        val_cls_loss = val_cls_loss / count if count > 0 else 0
        val_acc = val_acc / count if count > 0 else 0
        mean_dice = total_dice_fg / count if count > 0 else 0

        # log to wandb
        wandb.log({
            "val/seg_loss": val_seg_loss,
            "val/cls_loss": val_cls_loss,
            "val/cls_acc": val_acc,
            "val/dice_fg": mean_dice
        })

        self.print_to_log_file(
            f"Validation seg_loss={val_seg_loss:.4f}, cls_loss={val_cls_loss:.4f}, "
            f"acc={val_acc:.4f}, dice_fg={mean_dice:.4f}"
        )
        return val_seg_loss