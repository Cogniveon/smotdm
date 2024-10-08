import einops
import torch
from torchmetrics import Metric


class MeanStdMetric(Metric):
    def __init__(self, nfeats=166, max_len=600):
        super().__init__()
        self.nfeats = nfeats
        self.max_len = max_len
        self.add_state(
            "sums", default=torch.zeros(self.max_len, self.nfeats), dist_reduce_fx="sum"
        )
        self.add_state(
            "sum_of_squares",
            default=torch.zeros(self.max_len, self.nfeats),
            dist_reduce_fx="sum",
        )
        self.add_state("count", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, feature_tensors: torch.Tensor, count: int) -> None:
        if feature_tensors.shape[-1] != self.nfeats:
            raise ValueError("Feature dim does not match!")

        # Check if padding is needed
        current_len = feature_tensors.shape[1]
        if current_len < self.max_len:
            padding = torch.zeros(
                (self.max_len - current_len, self.nfeats), device=feature_tensors.device
            )
            padding_repeated = einops.repeat(
                padding, "x f -> b x f", b=feature_tensors.shape[0]
            )
            feature_tensors = torch.cat([feature_tensors, padding_repeated], dim=1)

        self.count += count
        self.sums += feature_tensors.sum(dim=0)
        self.sum_of_squares += (feature_tensors**2).sum(dim=0)

    def compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.sums / self.count
        variance = (self.sum_of_squares / self.count) - (mean**2)
        std = torch.sqrt(variance)
        return mean, std
