import torch
import torch.nn as nn
import math
import torch.nn.functional as F

import pdb


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


class HingeMLoss(torch.nn.Module):
    """
    Hinge margin loss function.
    Based on: https://arxiv.org/pdf/1712.01907.pdf
    """

    def __init__(self, margin_push=5.0, margin_pull=1.0):
        super(HingeMLoss, self).__init__()
        self.margin_push = margin_push      # label=1
        self.margin_pull = margin_pull      # label=0

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_hinge = torch.mean(
                            (1-label) * torch.pow(torch.clamp(euclidean_distance - self.margin_pull, min=0.0), 2) +
                            (label)   * torch.pow(torch.clamp(self.margin_push - euclidean_distance, min=0.0), 2)
                        )


        return loss_hinge