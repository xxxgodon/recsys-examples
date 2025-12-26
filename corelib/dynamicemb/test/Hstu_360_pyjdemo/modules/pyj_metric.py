import torch
from torchmetrics import AUROC
from torchmetrics.metric import Metric

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