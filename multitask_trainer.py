import torch
import torch.nn as nn
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer
from model.classification_head import ClassificationHead
import torch.nn.functional as F

class MultiTaskTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset)
        self.ce_loss = nn.CrossEntropyLoss()
        self.num_classes_cls = 3 
        self.classification_head = None
        self.total_loss_weight = 1.0

    def initialize(self):
        super().initialize()
        enc_channels = list(self.network.encoder.output_channels.values())[-1]
        self.classification_head = ClassificationHead(enc_channels, self.num_classes_cls).to(self.device)

    def forward(self, x):
        seg_output = self.network(x)                    
        features = self.network.encoder.get_output()    
        logits = self.classification_head(features[-1]) 
        return seg_output, logits

    def compute_loss(self, seg_output, target_seg, logits, subtype_labels):
        dice_loss = self.loss(seg_output, target_seg)
        ce_loss = self.ce_loss(logits, subtype_labels)
        return dice_loss + ce_loss * self.total_loss_weight

    def run_training_iteration(self, data_generator):
        data_dict = next(data_generator)
        data = data_dict['data']
        target_seg = data_dict['target']
        subtype_labels = data_dict['subtype_labels'].long().to(self.device)

        self.optimizer.zero_grad()
        seg_output, logits = self.forward(data)
        loss = self.compute_loss(seg_output, target_seg, logits, subtype_labels)
        loss.backward()
        self.optimizer.step()

        self.print_to_log_file(f"Loss: {loss.item():.4f}")
        return loss.detach().cpu().numpy()