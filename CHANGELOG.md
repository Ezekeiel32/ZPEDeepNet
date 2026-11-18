# Changelog

Versions follow `MAJOR.MINOR.PATCH`.

## Unreleased

## v0.1.0

First public release.

- `AdaptiveZPENoise`: a noise generator with a learnable scale and bias, trained to a target mean and standard deviation.
- Training script with plots of the training history, the error curves and the final distribution.
- Smoke tests: output shape, reported statistics, and a short training run that must move toward the targets.
