# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import torch.utils._pytree as pytree

from torchrl.data import Composite, TensorSpec
from torchrl.modules import ProbabilisticActor
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as TDMod,
    TensorDictSequential as TDSeq,
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union, Tuple
from collections import OrderedDict

from active_adaptation.learning.modules import IndependentNormal, VecNorm
from active_adaptation.learning.utils.opt import OptimizerGroup
from active_adaptation.learning.ppo.common import *
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
torch.set_float32_matmul_precision('high')

import active_adaptation
import torch.distributed as distr
from torch.nn.parallel import DistributedDataParallel as DDP
from .encoders import MixedEncoder

@dataclass
class PPOConfig:
    _target_: str = f"{__package__}.ppo_atec_ext.PPOPolicy"
    name: str = "ppo_atec_ext"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 5e-4
    desired_kl: Union[float, None] = None
    clip_param: float = 0.2
    entropy_coef: float = 0.002

    muon: bool = False
    compile: bool = False
    use_ddp: bool = True

    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, "extero")
    stages: Tuple[str, ...] = ("policy", "prior")

    prior_horizon: int = 32
    prior_lr: float = 5e-4
    prior_inner_steps: int = 4


cs = ConfigStore.instance()
cs.store("ppo_atec_ext", node=PPOConfig(stages=("policy",)), group="algo")
cs.store("ppo_atec_ext_prior", node=PPOConfig(
    train_every=48,
    in_keys=("policy", "extero", "root_state_w"),
    stages=("prior",),
), group="algo")


class CVAE(nn.Module):
    """
    Conditional VAE over future trajectory given the same state as the policy.

    Generative story: sample ``z ~ p(z | c)``, then ``x ~ p(x | z, c)`` where ``c`` is the
    mixed proprio + extero embedding (same encoder as the policy). The training objective is
    the ELBO using ``q(z | x, c)``: reconstruction plus KL ``KL(q(z|x,c) || p(z|c))``.

    **Shapes**

    - ``proprio``: ``(B, D)`` with ``D = proprio_shape`` (last observation dim).
    - ``extero``: ``(B, C, H, W)`` matching ``extero_shape``.
    - ``future_trajectory``: ``(B, horizon, 6)`` or flattened ``(B, horizon * 6)``. Each step is
      3D position plus 3D heading direction, in the robot frame relative to the current root.

    The posterior **does** depend on the future: ``future_encoder`` maps the flattened trajectory to
    ``hidden_dim``, then ``posterior_net`` maps ``[condition(c), future_embedding]`` to ``q(z|x,c)``.
    (Concatenating raw ``x`` into one MLP would still be a valid encoder; a separate trajectory MLP
    scales better with horizon and aligns future features with the condition dimension.)

    ``encode_prior`` returns parameters of **p(z|c)** (for sampling latents without a target trajectory).
    ``decode`` maps ``(z, c)`` to the mean trajectory; use with ``z`` sampled from the prior during
    rollouts, or from the posterior inside ``compute_loss``.
    """

    def __init__(
        self,
        proprio_shape: torch.Size,
        extero_shape: torch.Size,
        horizon: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.trajectory_dim = horizon * 6

        self.condition_encoder = MixedEncoder(proprio_shape, extero_shape)

        def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.Mish(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, out_dim),
            )

        self.future_encoder = mlp(self.trajectory_dim, hidden_dim)
        self.prior_net = mlp(hidden_dim, 2 * latent_dim)
        self.posterior_net = mlp(2 * hidden_dim, 2 * latent_dim)
        self.decoder_net = mlp(hidden_dim + latent_dim, self.trajectory_dim)

    def _split_gaussian_params(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = h.chunk(2, dim=-1)
        return mu, logvar.clamp(min=-20.0, max=2.0)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_prior(
        self, proprio: torch.Tensor, extero: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prior ``p(z | c)``: returns ``(mu, logvar)`` each ``(B, latent_dim)``."""
        c = self.condition_encoder(proprio, extero)
        return self._split_gaussian_params(self.prior_net(c))

    def encode_posterior(
        self,
        proprio: torch.Tensor,
        extero: torch.Tensor,
        future_trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Posterior ``q(z | x, c)``: returns ``(mu, logvar)`` each ``(B, latent_dim)``."""
        c = self.condition_encoder(proprio, extero)
        x = future_trajectory.reshape(future_trajectory.shape[0], self.trajectory_dim)
        h_x = self.future_encoder(x)
        h = torch.cat([c, h_x], dim=-1)
        return self._split_gaussian_params(self.posterior_net(h))

    def decode(self, z: torch.Tensor, proprio: torch.Tensor, extero: torch.Tensor) -> torch.Tensor:
        """Decoder mean ``p(x | z, c)``: returns shape ``(B, horizon, 6)``."""
        c = self.condition_encoder(proprio, extero)
        h = torch.cat([c, z], dim=-1)
        x_hat = self.decoder_net(h)
        return x_hat.view(x_hat.shape[0], self.horizon, 6)

    def compute_loss(
        self,
        proprio: torch.Tensor,  # (B, D)
        extero: torch.Tensor,  # (B, C, H, W)
        future_trajectory: torch.Tensor,  # (B, horizon, 6)
        beta_kl: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-sample ELBO (per batch mean).

        Returns:
            ``total``: ``recon + beta_kl * kl`` (scalar tensor, backprop target).
            ``recon``: mean squared error between decoder mean and target trajectory.
            ``kl``: mean ``KL(q(z|x,c) || p(z|c))`` (before ``beta_kl``).
        """
        x = future_trajectory.reshape(future_trajectory.shape[0], self.horizon, 6)
        mu_q, logvar_q = self.encode_posterior(proprio, extero, x)
        mu_p, logvar_p = self.encode_prior(proprio, extero)
        z = self._reparameterize(mu_q, logvar_q)
        x_hat = self.decode(z, proprio, extero)

        recon = F.mse_loss(x_hat, x, reduction="mean")

        var_q = logvar_q.exp()
        var_p = logvar_p.exp()
        log_var_ratio = logvar_p - logvar_q
        mu_diff_sq = (mu_q - mu_p) ** 2
        kl = 0.5 * (
            log_var_ratio.sum(dim=-1)
            - self.latent_dim
            + (var_q / var_p).sum(dim=-1)
            + (mu_diff_sq / var_p).sum(dim=-1)
        )
        kl = kl.mean()
        total = recon + beta_kl * kl
        return total, recon, kl

    @torch.no_grad()
    def sample_future(
        self,
        proprio: torch.Tensor,
        extero: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Sample ``z ~ p(z|c)`` and return decoder mean trajectory, shape ``(B, num_samples, H, 6)``."""
        mu_p, logvar_p = self.encode_prior(proprio, extero)
        B = mu_p.shape[0]
        mu = mu_p.unsqueeze(1).expand(B, num_samples, self.latent_dim).reshape(B * num_samples, self.latent_dim)
        logvar = logvar_p.unsqueeze(1).expand(B, num_samples, self.latent_dim).reshape(
            B * num_samples, self.latent_dim
        )
        z = self._reparameterize(mu, logvar)
        proprio_e = proprio.unsqueeze(1).expand(B, num_samples, *proprio.shape[1:]).reshape(
            B * num_samples, *proprio.shape[1:]
        )
        extero_e = extero.unsqueeze(1).expand(B, num_samples, *extero.shape[1:]).reshape(
            B * num_samples, *extero.shape[1:]
        )
        out = self.decode(z, proprio_e, extero_e)
        return out.view(B, num_samples, self.horizon, 6)



class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: Composite, 
        action_spec: Composite, 
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__()
        self.cfg = PPOConfig(**cfg)
        self.device = device

        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = 1.0
        self.desired_kl = self.cfg.desired_kl
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.gae = GAE(0.99, 0.95)
        
        fake_input = observation_spec.zero()
        proprio_shape = fake_input[OBS_KEY].shape[-1]
        extero_shape = fake_input["extero"].shape[-3:]
        self.action_dim = env.action_manager.action_dim

        self.mlp_norm = VecNorm(proprio_shape, proprio_shape, 1.0)
        self.cnn_norm = VecNorm(extero_shape, [extero_shape[0], 1, 1], 1.0)
        
        self.vecnorm = TDSeq(
            TDMod(self.mlp_norm, [OBS_KEY], ["_obs_normed"]),
            TDMod(self.cnn_norm, ["extero"], ["_extero_normed"]),
        ).to(self.device)
        
        self.obs_transform = env.observation_funcs[OBS_KEY].symmetry_transform().to(self.device)
        self.extero_transform = env.observation_funcs["extero"].symmetry_transform().to(self.device)
        self.act_transform = env.input_managers[ACTION_KEY].symmetry_transform().to(self.device)

        _actor = nn.Sequential(ResidualFC(256, 256), Actor(self.action_dim))
        actor_module = TDSeq(
            TDMod(
                MixedEncoder(proprio_shape, extero_shape),
                ["_obs_normed", "_extero_normed"], ["actor_feature"]
            ),
            TDMod(_actor, ["actor_feature"], ["loc", "scale"])
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        _critic = nn.Sequential(ResidualFC(256, 256), nn.LazyLinear(1))
        self.critic = TDSeq(
            TDMod(
                MixedEncoder(proprio_shape, extero_shape),
                ["_obs_normed", "_extero_normed"], ["critic_feature"]
            ),
            TDMod(_critic, ["critic_feature"], ["state_value"])
        ).to(self.device)

        self.vecnorm(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.02)
                nn.init.constant_(module.bias, 0.)
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
            if isinstance(module, Actor):
                nn.init.orthogonal_(module.actor_mean.weight, 0.01)
                nn.init.constant_(module.actor_mean.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)

        self._configure_distributed()
        self._configure_optimizers()

        self.prior_horizon = self.cfg.prior_horizon
        self.cvae = CVAE(
            proprio_shape, extero_shape, horizon=self.prior_horizon
        ).to(self.device)
        with torch.no_grad():
            p0 = fake_input[OBS_KEY].reshape(-1, proprio_shape)
            e0 = fake_input["extero"].reshape(-1, *extero_shape)
            x0 = torch.zeros(
                p0.shape[0], self.prior_horizon, 6, device=self.device, dtype=p0.dtype
            )
            self.cvae.compute_loss(p0, e0, x0)[0]
        self.cvae.apply(init_)
        self.prior_opt = torch.optim.AdamW(
            self.cvae.parameters(), lr=self.cfg.prior_lr, weight_decay=0.01
        )

        self.update = self._update
        self.stage = self.cfg.stages[0]
    
    def _configure_optimizers(self):
        def is_matrix_shaped(param: torch.Tensor) -> bool:
            return param.dim() == 2

        if self.cfg.muon:
            muon = torch.optim.Muon([
                {"params": [p for p in self.actor.parameters() if is_matrix_shaped(p)]},
                {"params": [p for p in self.critic.parameters() if is_matrix_shaped(p)]},
            ], lr=self.cfg.lr, adjust_lr_fn="match_rms_adamw", weight_decay=0.01)

            adamw = torch.optim.AdamW([
                {"params": [p for p in self.actor.parameters() if not is_matrix_shaped(p)]},
                {"params": [p for p in self.critic.parameters() if not is_matrix_shaped(p)]},
            ], lr=self.cfg.lr, weight_decay=0.01)
            self.opt = OptimizerGroup([muon, adamw])
        else:
            self.opt = torch.optim.AdamW(
                [
                    {"params": self.actor.parameters()},
                    {"params": self.critic.parameters()},
                ],
                lr=self.cfg.lr,
                weight_decay=0.01
            )
    
    def _configure_distributed(self):
        if active_adaptation.is_distributed():
            if self.cfg.use_ddp:
                self.actor = DDP(self.actor)
                self.critic = DDP(self.critic)
                self.cvae = DDP(self.cvae)
            else:
                for param in self.actor.parameters():
                    distr.broadcast(param, src=0)
                for param in self.critic.parameters():
                    distr.broadcast(param, src=0)
                for param in self.cvae.parameters():
                    distr.broadcast(param, src=0)
        self.world_size = active_adaptation.get_world_size()
    
    def on_stage_start(self, stage: str):
        self.stage = stage

    def get_rollout_policy(self, mode: str="train", critic: bool=False):
        if critic:
            modules = [self.vecnorm, self.actor, self.critic]
        else:
            modules = [self.vecnorm, self.actor]
        if self.stage == "prior" and mode != "train":
            def foo(tensordict: TensorDict):
                proprio = self.mlp_norm(tensordict[OBS_KEY])
                extero = self.cnn_norm(tensordict["extero"])
                mu, _ = self.cvae.encode_prior(proprio, extero)
                future = self.cvae.decode(mu, proprio, extero)
                tensordict["future_traj"] = future[..., :3]
                return tensordict
            modules.append(foo)
        rollout_policy = TDSeq(*modules)
        return rollout_policy

    def train_op(self, tensordict: TensorDict):
        if self.stage == "policy":
            return self.train_policy(tensordict)
        elif self.stage == "prior":
            return self.train_prior(tensordict)
        else:
            raise ValueError(f"Invalid stage: {self.stage}")
    
    @VecNorm.freeze()
    def train_policy(self, tensordict: TensorDict):
        assert VecNorm.FROZEN, "VecNorm must be frozen before training"

        tensordict = tensordict.exclude("stats", ("next", "stats"))

        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret")
        action = tensordict[ACTION_KEY]
        adv_unnormalized = tensordict["adv"].clone()
        log_probs_before = tensordict["action_log_prob"]
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self.update(minibatch))
                
                if self.desired_kl is not None: # adaptive learning rate
                    kl = infos[-1]["actor/kl"]
                    actor_lr = self.opt.param_groups[0]["lr"]
                    if kl > self.desired_kl * 2.0:
                        actor_lr = max(1e-5, actor_lr / 1.5)
                    elif kl < self.desired_kl / 2.0 and kl > 0.0:
                        actor_lr = min(1e-2, actor_lr * 1.5)
                    self.opt.param_groups[0]["lr"] = actor_lr
        
        with torch.no_grad():
            tensordict_ = self.actor(tensordict.copy())
            dist = IndependentNormal(tensordict_["loc"], tensordict_["scale"])
            log_probs_after = dist.log_prob(action)
            pg_loss_after = log_probs_after.reshape_as(adv_unnormalized) * adv_unnormalized
            pg_loss_before = log_probs_before.reshape_as(adv_unnormalized) * adv_unnormalized
                
        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.opt.param_groups[0]["lr"]
        infos["actor/pg_loss_raw_after"] = pg_loss_after.mean().item()
        infos["actor/pg_loss_raw_before"] = pg_loss_before.mean().item()

        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_var"] = tensordict["ret"].var().item()
        infos["critic/neg_rew_ratio"] = (tensordict[REWARD_KEY].sum(-1) <= 0.).float().mean().item()
        if active_adaptation.is_distributed():
            self.mlp_norm.synchronize(mode="broadcast")
            self.cnn_norm.synchronize(mode="broadcast")
        return dict(sorted(infos.items()))

    @VecNorm.freeze()
    def train_prior(self, tensordict: TensorDict):
        assert VecNorm.FROZEN, "VecNorm must be frozen before training"
        N, T = tensordict.shape[:2]
        H = self.prior_horizon
        device = tensordict.device
        dtype = tensordict[OBS_KEY].dtype

        Sl = T - H
        t_idx = torch.arange(Sl, device=device)
        offsets = torch.arange(1, H + 1, device=device)
        ft = t_idx.view(Sl, 1) + offsets.view(1, H) # (Sl, H)

        # filter out cross-episode trajectories
        has_done = tensordict[DONE_KEY].any(dim=1).reshape(N)

        root_state_w = tensordict["root_state_w"][~has_done]
        root_pos_w, root_quat_w, _, _ = root_state_w.split([3, 4, 3, 3], dim=-1)
        pos_future_w = root_pos_w[:, ft, :]
        quat_future_w = root_quat_w[:, ft, :]
        quat_cur = root_quat_w[:, :Sl, None, :]
        pos_cur = root_pos_w[:, :Sl, None, :]
        delta_w = pos_future_w - pos_cur
        
        rel_pos_b = quat_rotate_inverse(quat_cur, delta_w)
        ex = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype).view(1, 1, 1, 3)
        heading_w = quat_rotate(quat_future_w, ex)
        heading_b = quat_rotate_inverse(quat_cur, heading_w)
        future_traj = torch.cat([rel_pos_b, heading_b], dim=-1)

        obs_proprio = tensordict[OBS_KEY][~has_done, :Sl] # (N, T, D)
        obs_extero = tensordict["extero"][~has_done, :Sl] # (N, T, C, H, W)
        
        proprio = self.mlp_norm(obs_proprio).flatten(0, 1) # (N * Sl, D)
        extero = self.cnn_norm(obs_extero).flatten(0, 1) # (N * Sl, C, H, W)
        future_flat = future_traj.flatten(0, 1) # (N * Sl, horizon, 6)
        
        infos = {"prior/loss": [], "prior/recon_loss": [], "prior/kl_loss": []}
        batch_indices = torch.arange(proprio.shape[0], device=device).reshape(self.cfg.prior_inner_steps, -1)
        beta = 1.0
        for i in batch_indices.unbind(0):
            self.prior_opt.zero_grad(set_to_none=True)
            loss, recon, kl = self.cvae.compute_loss(
                proprio[i], extero[i], future_flat[i], beta_kl=beta
            )
            loss.backward()
            if active_adaptation.is_distributed() and not self.cfg.use_ddp:
                for param in self.cvae.parameters():
                    distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                    param.grad /= self.world_size
            nn.utils.clip_grad_norm_(self.cvae.parameters(), self.max_grad_norm)
            self.prior_opt.step()

            infos["prior/loss"].append(loss.detach().item())
            infos["prior/recon_loss"].append(recon.detach().item())
            infos["prior/kl_loss"].append(kl.detach().item())

        for k in ("prior/loss", "prior/recon_loss", "prior/kl_loss"):
            infos[k] = sum(infos[k]) / len(infos[k])

        if active_adaptation.is_distributed():
            self.mlp_norm.synchronize(mode="broadcast")
            self.cnn_norm.synchronize(mode="broadcast")

        return dict(sorted(infos.items()))

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: TDMod, 
        adv_key: str="adv",
        ret_key: str="ret",
    ):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                critic(self.vecnorm(tensordict_flat))
                critic(self.vecnorm(tensordict_flat["next"]))

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True)# .clamp_min(0.)
        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]

        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _update(self, tensordict: TensorDict):
        bsize = tensordict.shape[0]
        loc_old, scale_old = tensordict["loc"], tensordict["scale"]

        symmetry = tensordict.empty()
        symmetry[ACTION_KEY] = self.act_transform(tensordict[ACTION_KEY])
        symmetry[OBS_KEY] = self.obs_transform(tensordict[OBS_KEY])
        symmetry["extero"] = self.extero_transform(tensordict["extero"])
        symmetry["action_log_prob"] = tensordict["action_log_prob"]
        symmetry["adv"] = tensordict["adv"]
        symmetry["ret"] = tensordict["ret"]
        symmetry["is_init"] = tensordict["is_init"]
        tensordict = torch.cat([tensordict.select(*symmetry.keys(True, True)), symmetry], dim=0)
        
        self.vecnorm(tensordict)

        valid = (~tensordict["is_init"])
        valid_cnt = valid.sum()
        
        action_data = tensordict[ACTION_KEY]
        log_probs_data = tensordict["action_log_prob"]
        self.actor(tensordict)
        dist = IndependentNormal(tensordict["loc"], tensordict["scale"])
        log_probs = dist.log_prob(action_data)
        entropy = (dist.entropy().reshape_as(valid) * valid).sum() / valid_cnt

        adv = tensordict["adv"]
        log_ratio = (log_probs - log_probs_data).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - (torch.min(surr1, surr2).reshape_as(valid) * valid).sum() / valid_cnt
        entropy_loss = - self.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss.reshape_as(valid) * valid).sum() / valid_cnt

        loss = policy_loss + entropy_loss + value_loss
        self.opt.zero_grad()
        loss.backward()

        if active_adaptation.is_distributed() and not self.cfg.use_ddp:
            for param in self.actor.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size
            for param in self.critic.parameters():
                distr.all_reduce(param.grad, op=distr.ReduceOp.SUM)
                param.grad /= self.world_size

        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt.step()
        
        with torch.no_grad():
            explained_var = 1 - value_loss / b_returns[valid].var()
            clipfrac = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            loc, scale = dist.loc[:bsize], dist.scale[:bsize]
            kl = IndependentNormal.kl(loc, scale, loc_old, scale_old).mean()
            symmetry_loss = F.mse_loss(dist.mean[bsize:], self.act_transform(dist.mean[:bsize]))
        return {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/grad_norm": actor_grad_norm,
            "actor/clamp_ratio": clipfrac,
            "actor/kl": kl,
            "actor/symmetry_loss": symmetry_loss.detach(),
            "critic/value_loss": value_loss.detach(),
            "critic/grad_norm": critic_grad_norm,
            "critic/explained_var": explained_var,
        }

    def state_dict(self):
        state_dict = OrderedDict()
        for name, module in self.named_children():
            if isinstance(module, DDP):
                module = module.module
            state_dict[name] = module.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        succeed_keys = []
        failed_keys = []
        for name, module in self.named_children():
            _state_dict = state_dict.get(name, {})
            try:
                if isinstance(module, DDP):
                    module = module.module
                module.load_state_dict(_state_dict, strict=strict)
                succeed_keys.append(name)
            except Exception as e:
                warnings.warn(f"Failed to load state dict for {name}: {str(e)}")
                failed_keys.append(name)
        print(f"Successfully loaded {succeed_keys}.")
        return failed_keys


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)
