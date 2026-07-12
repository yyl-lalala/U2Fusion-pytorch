# ewc.py
import torch

class EWC:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = {}
        self.star = {}

    def compute_fisher(self, dataloader, loss_fn, num_samples=200):
        # loss_fn(S1, S2) 返回标量损失，不含EWC项
        for name in self.params:
            self.fisher[name] = torch.zeros_like(self.params[name], device=self.device)
        self.model.eval()
        samples = 0
        for batch in dataloader:
            if samples >= num_samples:
                break
            S1 = batch['S1'].to(self.device)
            S2 = batch['S2'].to(self.device)
            self.model.zero_grad()
            loss = loss_fn(S1, S2)
            loss.backward()
            for name, p in self.params.items():
                if p.grad is not None:
                    self.fisher[name] += p.grad.data ** 2
            samples += S1.size(0)
        for name in self.fisher:
            self.fisher[name] /= samples

    def update_star(self):
        for name, p in self.params.items():
            self.star[name] = p.data.clone().to(self.device)

    def ewc_loss(self, lam):
        loss = 0.0
        for name, p in self.params.items():
            if name in self.star and name in self.fisher:
                # 将star权重也放在同一个device
                star = self.star[name].to(self.device)
                fisher = self.fisher[name].to(self.device)
                loss += (fisher * (p - star) ** 2).sum()
        return lam / 2 * loss