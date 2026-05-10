import torch


def kullback_leibler_loss(mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
    kl = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp())
    return kl.sum(dim=1).mean()
