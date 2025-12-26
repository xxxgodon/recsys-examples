import torch
from torchmetrics import AUROC
from torchmetrics.metric import Metric

class CustomAUC(Metric):
    def __init__(self, task="binary", num_classes=None, **kwargs):
        super().__init__(**kwargs)
        # 支持 binary / multiclass / multilabel；根据 task 自动选择
        self.auroc = AUROC(task=task, num_classes=num_classes, **kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # preds: (N,) or (N, C) logits/probs；target: (N,) int labels or (N, C) bool/multi-label
        self.auroc.update(preds, target)

    def compute(self) -> torch.Tensor:
        return self.auroc.compute()

    def reset(self):
        self.auroc.reset()


class CustomCOPC(Metric):
    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        # 状态变量：累加 total clicks 和 total predicted clicks
        self.add_state("total_clicks", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_pred_clicks", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds: (N,) or (N, 1) —— 模型输出的 click probability 或 predicted click count
            target: (N,) or (N, 1) —— ground truth clicks (0/1 or float count)
        """
        if preds.ndim > 1 and preds.shape[1] == 1:
            preds = preds.squeeze(-1)
        if target.ndim > 1 and target.shape[1] == 1:
            target = target.squeeze(-1)

        # 支持 float target（如平滑 label、sampled count）
        self.total_clicks += target.sum()
        self.total_pred_clicks += preds.sum()

    def compute(self) -> torch.Tensor:
        """Returns COPC = sum(target) / sum(preds)"""
        return self.total_clicks / (self.total_pred_clicks + self.eps)

    def reset(self):
        self.total_clicks.zero_()
        self.total_pred_clicks.zero_()


auc_metric = CustomAUC()
copc_metric = CustomCOPC()

sumcum_sample_n = 0
for batch in dataloader:
    x, y = batch
    preds, logits = model(x)
    # other codes
    # ...
    auc_metric.update(preds, y)
    copc_metric.update(preds,y)
    sumcum_sample_n += batch_size

    if sumcum_sample_n == 1000000:
        final_auc = auc_metric.compute()
        final_copc = copc_metric.compute()
        print(f"Validation AUC: {final_auc:.4f}, COPC: {final_copc:.4f}")
        auc_metric.reset()
        copc_metric.reset()
        sumcum_sample_n = 0

final_auc = auc_metric.compute()
print(f"Validation AUC: {final_auc:.4f}")
auc_metric.reset()  # 下一轮前重置