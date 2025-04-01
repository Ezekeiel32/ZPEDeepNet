# ZPE-Deep-Net
An adaptive Zero-Point Energy (ZPE) noise generator built with PyTorch, designed to produce quantum noise with a target standard deviation. This project simulates ZPE fluctuations for quantum systems, aiming for a specific noise variance (e.g., std=0.05).

## Features
- Generates Gaussian noise with a target standard deviation.
- Uses PyTorch for tensor operations and optimization.
- Includes training loop to adaptively adjust noise parameters.
- Visualizes training progress with matplotlib plots.

## Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/ezekeiel32/ZPEDeepNet.git
   cd ZPE-Quantum-Noise

2. Install dependencies:
bash
pip install -r requirements.txt

3. Run the code:
bash

python zpe_noise.py

Usage:

The script trains an AdaptiveZPENoise model to generate noise with a target standard deviation of 0.05 over 1000 steps. 
It outputs logs and plots showing the noise mean, scale, bias, and clamp factor over training.

Results:

After 1000 training steps, the noise standard deviation stabilizes around 0.05, with a random, Gaussian distribution centered near 0.


License:

MIT License
