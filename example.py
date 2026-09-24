"""Run one forward and backward pass on synthetic inputs."""
import torch
from loadmixer import LoadMixer

if __name__ == "__main__":
    torch.manual_seed(0)
    model = LoadMixer()
    history = torch.randn(2, 96, 1)
    calendar = torch.rand(2, 96, 5) - 0.5
    prediction = model(history, calendar)
    target = torch.randn(2, 12, 1)
    (prediction - target).square().mean().backward()
    print("Prediction shape:", tuple(prediction.shape))
    print("Forward and backward passes succeeded.")
