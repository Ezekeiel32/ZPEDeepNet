import torch
import math
import matplotlib.pyplot as plt

print(f"PyTorch Version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class AdaptiveZPENoise(torch.nn.Module):
    def __init__(self, dim_size, h_bar=1.0, omega=1.0, initial_clamp=0.01, 
                 min_std=1e-6, learn_rate=0.001):
        super(AdaptiveZPENoise, self).__init__()
        self.E0 = 0.5 * h_bar * omega
        self.dim_size = dim_size
        
        initial_scale = self.E0 * 32.0
        for _ in range(5):
            initial_scale = initial_scale * 11.0 / 16.0
        initial_bias = self.E0 * 32.0 - initial_scale
        
        self.zpe_scale = torch.nn.Parameter(torch.ones(dim_size, device=device) * initial_scale)
        self.zpe_bias = torch.nn.Parameter(torch.ones(dim_size, device=device) * initial_bias)
        self.clamp_factor = initial_clamp
        
        self.min_std = min_std
        self.train_steps = 0
        self.noise_history = []
        self.scale_history = []
        self.bias_history = []
        self.clamp_history = []
        self.optimizer = torch.optim.Adam([self.zpe_scale, self.zpe_bias], lr=learn_rate)
        
    def forward(self, time_factor=None):
        if time_factor is None:
            time_factor = 1.0 + 0.1 * math.sin(self.train_steps * 0.1)
        
        effective_clamp = self.clamp_factor * (1.0 + 0.01 * self.train_steps)
        raw_noise = torch.randn(self.dim_size, device=device)
        scaled_noise = raw_noise * (effective_clamp * self.E0 * 1.0)  # Increased from 0.9
        
        dynamic_scale = self.zpe_scale * time_factor
        noise = scaled_noise * dynamic_scale
        
        noise = torch.clamp(noise, -effective_clamp * self.E0, effective_clamp * self.E0)
        return noise
    
    def train_step(self, target_std=None, target_mean=None):
        self.train_steps += 1
        noise = self.forward()
        
        loss = 0
        noise_std = noise.std()
        noise_mean = noise.mean()
        
        if target_std is not None:
            if noise_std < target_std:
                variance_penalty = 20.0 * (target_std - noise_std).pow(2)
                loss += variance_penalty
                # Slightly stronger push on scale
                loss += 0.2 * torch.nn.functional.mse_loss(self.zpe_scale.mean(), 
                                                           torch.tensor(target_std * 3.5, device=device))
            else:
                variance_penalty = 20.0 * (noise_std - target_std).pow(2)
                loss += variance_penalty
                loss += 0.1 * torch.nn.functional.mse_loss(self.zpe_scale.mean(), 
                                                           torch.tensor(target_std, device=device))
        
        if target_mean is not None:
            loss += torch.nn.functional.mse_loss(noise_mean, torch.tensor(target_mean, device=device))
            loss += 0.1 * torch.nn.functional.mse_loss(self.zpe_bias.mean(), 
                                                       torch.tensor(target_mean, device=device))
        
        if noise_std < self.min_std:
            loss += 1.0 * (self.min_std - noise_std)
        
        loss += 0.0001 * (self.zpe_scale.pow(2).mean() + self.zpe_bias.pow(2).mean())
        
        if loss > 0:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.noise_history.append(noise.detach().mean().cpu().item())
        self.scale_history.append(self.zpe_scale.detach().mean().cpu().item())
        self.bias_history.append(self.zpe_bias.detach().mean().cpu().item())
        self.clamp_history.append(self.clamp_factor * (1.0 + 0.01 * self.train_steps))
        
        return noise.detach(), loss.detach() if loss > 0 else torch.tensor(0.0)

    def get_noise_stats(self):
        with torch.no_grad():
            noise = self.forward()
            return {
                'mean': noise.mean().item(),
                'std': noise.std().item(),
                'min': noise.min().item(),
                'max': noise.max().item(),
                'clamp_factor': self.clamp_factor * (1.0 + 0.01 * self.train_steps),
                'scale_mean': self.zpe_scale.mean().item(),
                'bias_mean': self.zpe_bias.mean().item()
            }
            
    def plot_history(self):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(self.noise_history)
        plt.title('Noise Mean Over Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Mean Value')
        plt.subplot(2, 2, 2)
        plt.plot(self.scale_history)
        plt.title('Scale Parameter Over Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Scale Value')
        plt.subplot(2, 2, 3)
        plt.plot(self.bias_history)
        plt.title('Bias Parameter Over Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Bias Value')
        plt.subplot(2, 2, 4)
        plt.plot(self.clamp_history)
        plt.title('Clamp Factor Over Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Clamp Value')
        plt.tight_layout()
        plt.show()

class ToyModel(torch.nn.Module):
    def __init__(self, dim_size):
        super().__init__()
        self.linear = torch.nn.Linear(dim_size, dim_size)
    
    def forward(self, x, noise):
        return self.linear(x + noise)

def train_zpe_noise(dim_size=512, target_std=0.05, training_steps=1000):
    print(f"\n=== Training Adaptive ZPE Noise ===")
    print(f"Dim Size: {dim_size}, Target Std: {target_std}")
    print(f"Training Steps: {training_steps}")
    print(f"Device: {device}")
    
    noise_gen = AdaptiveZPENoise(dim_size).to(device)
    
    stats = noise_gen.get_noise_stats()
    print(f"\nInitial Stats:")
    print(f"Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
    print(f"Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
    print(f"Clamp Factor: {stats['clamp_factor']:.6f}")
    
    for step in range(training_steps):
        noise, loss = noise_gen.train_step(target_std=target_std)
        if (step+1) % 100 == 0 or step == 0:
            stats = noise_gen.get_noise_stats()
            print(f"\nStep {step+1}/{training_steps}, Loss: {loss.item():.6f}")
            print(f"Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
            print(f"Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
            print(f"Clamp Factor: {stats['clamp_factor']:.6f}")
            print(f"Scale Mean: {stats['scale_mean']:.6f}, Bias Mean: {stats['bias_mean']:.6f}")
            print(f"Noise Sample (first 5): {noise[:5].tolist()}")
    
    stats = noise_gen.get_noise_stats()
    print(f"\n=== Final Stats ===")
    print(f"Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
    print(f"Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
    print(f"Clamp Factor: {stats['clamp_factor']:.6f}")
    
    noise_gen.plot_history()
    return noise_gen

if __name__ == "__main__":
    dim_size = 512
    target_std = 0.05
    training_steps = 1000
    
    noise_gen = train_zpe_noise(dim_size, target_std, training_steps)
    
    print("\n=== Integrating with Toy Model ===")
    model = ToyModel(dim_size).to(device)
    dummy_data = torch.zeros(1, dim_size).to(device)
    output = model(dummy_data, noise_gen.forward())
    print(f"Output Sample (first 5): {output[0, :5].tolist()}")
