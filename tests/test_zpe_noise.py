"""Smoke tests: the generator returns one value per dimension, and training moves it toward its targets."""
import torch

import zpe_noise
from zpe_noise import AdaptiveZPENoise, ToyModel

TARGET_MEAN, TARGET_STD = 5.39, 0.1


def make_generator(dim=64):
    torch.manual_seed(0)
    return AdaptiveZPENoise(dim, target_mean=TARGET_MEAN, target_std=TARGET_STD).to(zpe_noise.device)


def test_forward_returns_one_finite_value_per_dimension():
    noise = make_generator(64).forward()
    assert noise.shape[-1] == 64
    assert torch.isfinite(noise).all()


def test_stats_report_the_expected_fields():
    stats = make_generator().get_noise_stats()
    assert {"mean", "std", "min", "max", "scale_mean", "bias_mean"} <= set(stats)


def test_training_moves_the_noise_toward_its_targets():
    generator = make_generator()
    before = generator.get_noise_stats()
    for _ in range(300):
        generator.train_step()
    after = generator.get_noise_stats()
    assert abs(after["mean"] - TARGET_MEAN) < 0.05
    assert abs(after["std"] - TARGET_STD) < abs(before["std"] - TARGET_STD)


def test_toy_model_accepts_the_noise():
    generator = make_generator(32)
    model = ToyModel(32).to(zpe_noise.device)
    output = model(torch.zeros(1, 32, device=zpe_noise.device), generator.forward())
    assert torch.isfinite(output).all()
