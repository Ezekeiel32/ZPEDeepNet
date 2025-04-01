import torch
import math
import matplotlib.pyplot as plt
import os

print(f"PyTorch Version: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class AdaptiveZPENoise(torch.nn.Module):
    def __init__(self, dim_size, h_bar=1.0, omega=1.0, initial_clamp=0.1, 
                 min_std=1e-6, learn_rate=0.003, bias_learn_rate=0.008,
                 target_mean=5.39, target_std=0.1):
        super(AdaptiveZPENoise, self).__init__()
        self.E0 = 0.5 * h_bar * omega
        self.dim_size = dim_size
        self.target_mean = target_mean
        self.target_std = target_std
        
        initial_scale = target_std / (initial_clamp * self.E0 * 2.0)
        initial_bias = target_mean
        
        self.zpe_scale = torch.nn.Parameter(torch.ones(dim_size, device=device) * initial_scale)
        self.zpe_bias = torch.nn.Parameter(torch.ones(dim_size, device=device) * initial_bias)
        self.clamp_factor = initial_clamp
        
        self.min_std = min_std
        self.train_steps = 0
        self.noise_history = []
        self.scale_history = []
        self.bias_history = []
        self.clamp_history = []
        self.std_history = []
        
        self.optimizer = torch.optim.Adam([
            {'params': self.zpe_scale, 'lr': learn_rate},
            {'params': self.zpe_bias, 'lr': bias_learn_rate}
        ], betas=(0.5, 0.9))
        
    def forward(self, time_factor=None):
        if time_factor is None:
            time_factor = 1.0
        
        effective_clamp = self.clamp_factor * (1.0 + 0.001 * self.train_steps)
        effective_clamp = min(effective_clamp, 0.3)
        
        raw_noise = torch.randn(self.dim_size, device=device)
        scaled_noise = raw_noise * (effective_clamp * self.E0)
        
        dynamic_scale = torch.abs(self.zpe_scale) * time_factor
        noise = scaled_noise * dynamic_scale
        
        noise = noise + self.zpe_bias
        
        noise = torch.clamp(noise, self.target_mean - effective_clamp * self.E0, self.target_mean + effective_clamp * self.E0)
        
        if self.train_steps % 100 == 0 or self.train_steps < 5:
            print(f"Step {self.train_steps}: raw_noise std: {raw_noise.std().item():.6f}")
            print(f"Step {self.train_steps}: scaled_noise std: {scaled_noise.std().item():.6f}")
            print(f"Step {self.train_steps}: dynamic_scale mean: {dynamic_scale.mean().item():.6f}")
            print(f"Step {self.train_steps}: noise before bias std: {noise.std().item():.6f}")
            print(f"Step {self.train_steps}: zpe_bias mean: {self.zpe_bias.mean().item():.6f}")
        
        return noise
    
    def train_step(self, target_std=None, target_mean=None):
        self.train_steps += 1
        
        if target_std is None:
            target_std = self.target_std
        if target_mean is None:
            target_mean = self.target_mean
            
        noise = self.forward()
        
        loss = 0
        noise_std = noise.std()
        noise_mean = noise.mean()
        
        mean_diff = noise_mean - target_mean
        mean_loss = 20.0 * mean_diff.pow(2)
        loss += mean_loss
        
        std_diff = noise_std - target_std
        std_loss = 5.0 * std_diff.pow(2)
        loss += std_loss
        
        bias_diff = self.zpe_bias.mean() - target_mean
        bias_loss = 2.0 * bias_diff.pow(2)
        loss += bias_loss
        
        effective_clamp = self.clamp_factor * (1.0 + 0.001 * self.train_steps)
        effective_clamp = min(effective_clamp, 0.3)
        target_scale = target_std / (effective_clamp * self.E0 * 0.99)
        scale_diff = self.zpe_scale.mean() - target_scale
        scale_loss = 1.0 * scale_diff.pow(2)
        loss += scale_loss
        
        if noise_std < self.min_std:
            loss += 2.0 * (self.min_std - noise_std)
        
        loss += 0.00001 * (self.zpe_scale.pow(2).mean() + (self.zpe_bias - target_mean).pow(2).mean())
        
        if loss > 0:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
            
            with torch.no_grad():
                current_bias_mean = self.zpe_bias.mean().item()
                if abs(current_bias_mean - target_mean) > 0.1:
                    correction = 0.1 * (target_mean - current_bias_mean)
                    self.zpe_bias.add_(correction)
        
        self.noise_history.append(noise.detach().mean().cpu().item())
        self.scale_history.append(self.zpe_scale.detach().mean().cpu().item())
        self.bias_history.append(self.zpe_bias.detach().mean().cpu().item())
        self.clamp_history.append(min(self.clamp_factor * (1.0 + 0.001 * self.train_steps), 0.3))
        self.std_history.append(noise_std.detach().cpu().item())
        
        return noise.detach(), loss.detach() if loss > 0 else torch.tensor(0.0)

    def get_noise_stats(self):
        with torch.no_grad():
            noise = self.forward()
            return {
                'mean': noise.mean().item(),
                'std': noise.std().item(),
                'min': noise.min().item(),
                'max': noise.max().item(),
                'clamp_factor': min(self.clamp_factor * (1.0 + 0.001 * self.train_steps), 0.3),
                'scale_mean': self.zpe_scale.mean().item(),
                'bias_mean': self.zpe_bias.mean().item(),
                'target_mean': self.target_mean,
                'target_std': self.target_std
            }
            
    def plot_history(self):
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 2, 1)
        plt.plot(self.noise_history)
        plt.axhline(y=self.target_mean, color='r', linestyle='--', alpha=0.7, label=f'Target: {self.target_mean}')
        plt.title('Noise Mean Over Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Mean Value')
        plt.legend()
        
        plt.subplot(3, 2, 2)
        plt.plot(self.std_history)
        plt.axhline(y=self.target_std, color='r', linestyle='--', alpha=0.7, label=f'Target: {self.target_std}')
        plt.title('Noise Std Over Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Std Value')
        plt.legend()
        
        plt.subplot(3, 2, 3)
        plt.plot(self.scale_history)
        plt.title('Scale Parameter Over Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Scale Value')
        
        plt.subplot(3, 2, 4)
        plt.plot(self.bias_history)
        plt.axhline(y=self.target_mean, color='r', linestyle='--', alpha=0.7, label=f'Target: {self.target_mean}')
        plt.title('Bias Parameter Over Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Bias Value')
        plt.legend()
        
        plt.subplot(3, 2, 5)
        plt.plot(self.clamp_history)
        plt.title('Clamp Factor Over Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Clamp Value')
        
        plt.subplot(3, 2, 6)
        mean_diffs = [abs(m - self.target_mean) for m in self.noise_history]
        std_diffs = [abs(s - self.target_std) for s in self.std_history]
        plt.plot(mean_diffs, label='Mean Error')
        plt.plot(std_diffs, label='Std Error')
        plt.title('Error vs Target Over Training')
        plt.xlabel('Training Steps')
        plt.ylabel('Absolute Error')
        plt.yscale('log')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('plots/training_history.png')
        plt.close()

class ToyModel(torch.nn.Module):
    def __init__(self, dim_size):
        super().__init__()
        self.linear = torch.nn.Linear(dim_size, dim_size)
    
    def forward(self, x, noise):
        return self.linear(x + noise)

def train_zpe_noise(dim_size=512, target_std=0.1, target_mean=5.39, training_steps=1500):
    print(f"\n=== Training Adaptive ZPE Noise ===")
    print(f"Dim Size: {dim_size}, Target Std: {target_std}, Target Mean: {target_mean}")
    print(f"Training Steps: {training_steps}")
    print(f"Device: {device}")
    
    os.makedirs("plots", exist_ok=True)
    
    noise_gen = AdaptiveZPENoise(dim_size, 
                                target_mean=target_mean, 
                                target_std=target_std).to(device)
    
    stats = noise_gen.get_noise_stats()
    print(f"\nInitial Stats:")
    print(f"Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
    print(f"Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
    print(f"Clamp Factor: {stats['clamp_factor']:.6f}")
    print(f"Scale Mean: {stats['scale_mean']:.6f}, Bias Mean: {stats['bias_mean']:.6f}")
    
    best_combined_error = float('inf')
    patience = 100
    patience_counter = 0
    
    errors = []
    
    for step in range(training_steps):
        noise, loss = noise_gen.train_step(target_std=target_std, target_mean=target_mean)
        
        with torch.no_grad():
            mean_error = abs(noise.mean().item() - target_mean)
            std_error = abs(noise.std().item() - target_std)
            combined_error = mean_error + std_error
            errors.append((mean_error, std_error, combined_error))
        
        if (step+1) % 100 == 0 or step == 0:
            stats = noise_gen.get_noise_stats()
            print(f"\nStep {step+1}/{training_steps}, Loss: {loss.item():.6f}")
            print(f"Mean: {stats['mean']:.6f} (target: {target_mean}, error: {mean_error:.6f})")
            print(f"Std: {stats['std']:.6f} (target: {target_std}, error: {std_error:.6f})")
            print(f"Combined Error: {combined_error:.6f}")
            print(f"Scale Mean: {stats['scale_mean']:.6f}, Bias Mean: {stats['bias_mean']:.6f}")
            
            if mean_error < 0.001 and std_error < 0.001:
                print(f"Target reached at step {step+1}!")
                break
        
        if combined_error < best_combined_error:
            best_combined_error = combined_error
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience and step > 300:
            print(f"\nEarly stopping at step {step+1}")
            break
    
    # Final mean adjustment using multiple samples
    with torch.no_grad():
        means = []
        for _ in range(10):  # Average over 10 samples to reduce noise
            noise = noise_gen.forward()
            means.append(noise.mean().item())
        mean_error = sum(means) / len(means) - target_mean
        noise_gen.zpe_bias.add_(-mean_error)
    
    plt.figure(figsize=(10, 6))
    mean_errors, std_errors, combined_errors = zip(*errors)
    plt.plot(mean_errors, label='Mean Error')
    plt.plot(std_errors, label='Std Error')
    plt.plot(combined_errors, label='Combined Error')
    plt.title('Errors During Training')
    plt.xlabel('Steps')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.legend()
    plt.savefig('plots/error_curves.png')
    plt.close()
    
    stats = noise_gen.get_noise_stats()
    print(f"\n=== Final Stats ===")
    print(f"Mean: {stats['mean']:.6f} (target: {target_mean}, error: {abs(stats['mean'] - target_mean):.6f})")
    print(f"Std: {stats['std']:.6f} (target: {target_std}, error: {abs(stats['std'] - target_std):.6f})")
    print(f"Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
    print(f"Scale Mean: {stats['scale_mean']:.6f}, Bias Mean: {stats['bias_mean']:.6f}")
    
    with torch.no_grad():
        final_noise = noise_gen.forward().cpu().numpy()
        plt.figure(figsize=(10, 6))
        plt.hist(final_noise, bins=50)
        plt.axvline(x=target_mean, color='r', linestyle='--', label=f'Target Mean: {target_mean}')
        plt.title('Final Noise Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.savefig('plots/final_distribution.png')
        plt.close()
    
    noise_gen.plot_history()
    return noise_gen

if __name__ == "__main__":
    dim_size = 512
    target_std = 0.1
    target_mean = 5.39
    training_steps = 1500
    
    noise_gen = train_zpe_noise(dim_size, target_std, target_mean, training_steps)
    
    print("\n=== Integrating with Toy Model ===")
    model = ToyModel(dim_size).to(device)
    dummy_data = torch.zeros(1, dim_size).to(device)
    output = model(dummy_data, noise_gen.forward())
    print(f"Output Sample (first 5): {output[0, :5].tolist()}")
    
    print("\n=== Noise Stability Test ===")
    means = []
    stds = []
    for i in range(10):
        with torch.no_grad():
            test_noise = noise_gen.forward()
            means.append(test_noise.mean().item())
            stds.append(test_noise.std().item())
    
    print(f"Mean of means: {sum(means)/len(means):.6f} (target: {target_mean})")
    print(f"Mean of stds: {sum(stds)/len(stds):.6f} (target: {target_std})")
    print(f"Variance of means: {torch.tensor(means).var().item():.6f}")
    print(f"Variance of stds: {torch.tensor(stds).var().item():.6f}")
