import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.data import LazyMemmapStorage
from tensordict.nn import (
    TensorDictModule as TDMod, 
    TensorDictSequential as TDSEq,
)
from tensordict import TensorDict
from dataclasses import dataclass, MISSING
from pathlib import Path
from hydra.core.config_store import ConfigStore
from active_adaptation.learning.modules.vecnorm import VecNorm
cs = ConfigStore.instance()

@dataclass
class SimpleModelCfg:
    _target_: str = f"SimpleModel"
    dataset_path: str = "/home/btx0424/lab51/active-adaptation/scripts/rollout/SiriusATEC/2025-11-30-04-23-47"
    lsgan: bool = True # whether to use LSGAN's discriminator loss
    disc_logit_reg: float = 0.01
    disc_grad_penalty: float = 10.0


cs.store(name="simple_model", node=SimpleModelCfg, group="model")

class WassersteinDiscriminator(nn.Module):
    def __init__(self, activation=nn.LeakyReLU, grad_penalty_weight=10.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LazyLinear(512),
            # nn.LayerNorm(256), 
            activation(), #nn.Dropout(0.05),
            nn.LazyLinear(256),
            # nn.LayerNorm(256),
            activation(), # nn.Dropout(0.05),
        )
        self.discriminator = nn.LazyLinear(1)
        self.grad_penalty_weight = grad_penalty_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(self.encoder(x))
    
    def compute_loss(self, pos_samples: torch.Tensor, neg_samples: torch.Tensor) -> dict:
        """
        Compute WGAN-GP discriminator loss.
        
        Args:
            pos_samples: Real samples (data_A)
            neg_samples: Fake samples (data_B)
        
        Returns:
            Total loss including gradient penalty
        """
        # Compute discriminator scores
        pos_scores = self(pos_samples)
        neg_scores = self(neg_samples)
        
        # WGAN loss: maximize D(real) - D(fake)
        # For minimization, we use: E[D(fake)] - E[D(real)]
        wgan_loss = neg_scores.mean() - pos_scores.mean()
        
        # Gradient penalty: compute on interpolated samples
        batch_size = pos_samples.shape[0]
        device = pos_samples.device
        
        # Sample random interpolation coefficients
        epsilon = torch.rand(batch_size, 1, device=device)
        
        # Create interpolated samples
        interpolated = epsilon * pos_samples + (1 - epsilon) * neg_samples
        interpolated.requires_grad_(True)
        
        # Compute discriminator output on interpolated samples
        interp_scores = self(interpolated)
        
        # Compute gradients of discriminator output w.r.t. interpolated samples
        grad_outputs = torch.ones_like(interp_scores)
        gradients = torch.autograd.grad(
            outputs=interp_scores,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Compute gradient penalty: (||gradient|| - 1)^2
        gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1.0) ** 2).mean()
        
        # Total loss
        loss = wgan_loss + self.grad_penalty_weight * gradient_penalty
        
        return {
            "loss": loss,
            "grad_penalty": gradient_penalty,
            "grad_norm": gradient_norm,
            "score_pos": pos_scores,
            "score_neg": neg_scores,
        }

class SimpleModel:
    def __init__(self, cfg: SimpleModelCfg, device: torch.device) -> None:
        self.cfg = SimpleModelCfg(**cfg)
        self.device = device

        dataset_path = Path(cfg.dataset_path)
        storage = LazyMemmapStorage(max_size=1, scratch_dir=dataset_path / "storage")
        storage.loads(dataset_path / "storage")
        self.dataset = storage._storage # do not use ReplayBuffer for now
        self.dataset_size = self.dataset.shape.numel() # [T, N]

        self.observation_dim = self.dataset[0]["amp_"].shape[-1]
        self.action_dim = self.dataset[0]["action"].shape[-1]
        
        self.discriminator = WassersteinDiscriminator().to(self.device)
        # self.discriminator()
        self.reward_normalizer = VecNorm(input_shape=(1,), stats_shape=(1,)).to(self.device)

        # def init_(module):
        #     if isinstance(module, nn.Linear):
        #         # nn.init.uniform_(module.weight, -1., 1.)
        #         nn.init.constant_(module.bias, 0.)
        
        # self.discriminator.apply(init_)

        # obs normalization
        self.mean = self.dataset["amp_"].mean(dim=(0, 1)).to(self.device)
        self.std = self.dataset["amp_"].std(dim=(0, 1)).to(self.device)

        self.opt = torch.optim.AdamW([
            {"params": self.discriminator.parameters()},
        ], lr=1e-4)

        self.update_counter = 0
    
    def sample_data(self, batch_size: int = 4096) -> TensorDict:
        indices = torch.randint(0, self.dataset_size, (batch_size,))
        T, N = self.dataset.shape
        row_indices = indices // N
        col_indices = indices % N
        return self.dataset["amp_"][row_indices, col_indices]

    def train_op(self, data: TensorDict) -> dict:
        """
        we train the discriminator to differentiate between two sets of data:
        data_A: pre-collected from one simulation environment
        data_B: online-collected from another simulation environment
        """
        numel = data.numel()
        # valid = (data.view(-1)["step_count"] > 1).squeeze(1)
        pos_samples = self.sample_data(batch_size=numel).to(self.device)
        pos_samples = (pos_samples - self.mean) / self.std
        
        neg_samples = data.view(-1)["amp_"].to(self.device)  # [batch_size, time_steps, obs_dim]
        neg_samples = (neg_samples - self.mean) / self.std
        
        if not neg_samples.shape[-1] == pos_samples.shape[-1]:
            raise ValueError(f"neg_samples.shape[-1] {neg_samples.shape[-1]} != pos_samples.shape[-1] {pos_samples.shape[-1]}")
        
        self.discriminator.train()
        loss_dict = self.discriminator.compute_loss(pos_samples, neg_samples)
        self.opt.zero_grad()
        loss_dict["loss"].backward()
        self.opt.step()

        with torch.no_grad():
            mean_score_pos = loss_dict["score_pos"].mean().item()
            mean_score_neg = loss_dict["score_neg"].mean().item()
            mean_grad_norm = loss_dict["grad_norm"].mean().item()

            self.discriminator.eval()
            reward = self.discriminator(neg_samples).reshape(*data.shape, 1)
            reward_normalized = self.reward_normalizer(reward)
            data["next", "reward"] = torch.cat([data["next", "reward"], 0.02 * 2.0 * reward_normalized], dim=-1)
        
        return {
            "discriminator/loss": loss_dict["loss"].item(),
            "discriminator/grad_penalty": loss_dict["grad_penalty"].item(),
            "discriminator/grad_norm": mean_grad_norm,
            "discriminator/score_pos": mean_score_pos,
            "discriminator/score_neg": mean_score_neg,
            "discriminator/reward": reward.mean().item(),
            "discriminator/reward_normalized": reward_normalized.mean().item(),
            "discriminator/reward_normalized_std": reward_normalized.std().item(),
        }

