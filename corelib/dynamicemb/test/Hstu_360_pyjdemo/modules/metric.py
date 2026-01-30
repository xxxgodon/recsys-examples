import torch
from torchmetrics import AUROC
from torchmetrics.metric import Metric

import numpy as np

class CustomAUC(Metric):
    def __init__(self, task="binary", num_classes=None, **kwargs):
        super().__init__(**kwargs)
        self.auroc = AUROC(task=task, num_classes=num_classes, **kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.auroc.update(preds, target)

    def compute(self) -> torch.Tensor:
        return self.auroc.compute()

    def reset(self):
        self.auroc.reset()

class MaskedAUC(Metric):
    def __init__(self, task="binary", num_classes=None, **kwargs):
        super().__init__(**kwargs)
        self.auroc = AUROC(
            task=task,
            num_classes=num_classes,
            **kwargs
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        valid: bool | torch.Tensor = True
    ):
        """
        valid:
          - bool: 当前 batch 是否真实
          - Tensor[batch]: 样本级 mask
        """

        if not torch.is_tensor(valid):
            if not valid:
                return
            valid = torch.ones_like(target, dtype=torch.bool)

        preds = preds.flatten()
        target = target.flatten()

        valid = valid.to(preds.device)

        preds = preds[valid]
        target = target[valid]

        if preds.numel() == 0:
            return

        self.auroc.update(preds, target)

    def compute(self):
        return self.auroc.compute()

    def reset(self):
        self.auroc.reset()

class CustomCOPC(Metric):
    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.add_state("total_clicks", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_pred_clicks", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.ndim > 1 and preds.shape[1] == 1:
            preds = preds.squeeze(-1)
        if target.ndim > 1 and target.shape[1] == 1:
            target = target.squeeze(-1)
        
        self.total_clicks += target.sum()
        self.total_pred_clicks += preds.sum()

    def compute(self) -> torch.Tensor:
        return self.total_clicks / (self.total_pred_clicks + self.eps)

    def reset(self):
        self.total_clicks.zero_()
        self.total_pred_clicks.zero_()

class StreamingAUC:
    def __init__(self, num_bins=1000):
        self.num_bins = num_bins
        self.reset()

    def reset(self):
        self.pos_hist = np.zeros(self.num_bins, dtype=np.float64)
        self.neg_hist = np.zeros(self.num_bins, dtype=np.float64)

    def update(self, preds, labels):
        preds = preds.detach().cpu().numpy().flatten()
        labels = labels.detach().cpu().numpy().flatten()
        
        # 量化成区间
        bins = np.floor(preds * (self.num_bins - 1)).astype(int)

        for b, y in zip(bins, labels):
            if y > 0.5:
                self.pos_hist[b] += 1
            else:
                self.neg_hist[b] += 1

    def compute(self):
        """基于桶统计得到近似 AUC"""
        cum_neg = 0
        auc = 0

        for b in range(self.num_bins):
            pos = self.pos_hist[b]
            neg = self.neg_hist[b]

            # 所有 neg 在前（pair）
            auc += pos * cum_neg

            # 相同桶内部，pos & neg 随机排序 → 贡献 0.5
            auc += pos * neg * 0.5

            cum_neg += neg

        total_pos = self.pos_hist.sum()
        total_neg = self.neg_hist.sum()

        if total_pos == 0 or total_neg == 0:
            return 0.5

        return auc / (total_pos * total_neg)

class StreamingCOPC(Metric):
    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

        self.add_state(
            "total_clicks",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )
        self.add_state(
            "total_pred_clicks",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum"
        )

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor | bool = True
    ):
        """
        valid:
          - True / False
          - or shape = [batch_size] 的 mask
        """

        if not torch.is_tensor(valid):
            if not valid:
                return
            valid = torch.ones_like(target, dtype=torch.bool)

        preds = preds.flatten()
        target = target.flatten()

        valid = valid.to(preds.device)

        preds = preds[valid]
        target = target[valid]

        if preds.numel() == 0:
            return

        self.total_clicks += target.sum()
        self.total_pred_clicks += preds.sum()

    def compute(self):
        return self.total_clicks / (self.total_pred_clicks + self.eps)

    def reset(self):
        self.total_clicks.zero_()
        self.total_pred_clicks.zero_()
