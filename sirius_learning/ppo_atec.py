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
import torch.distributions as D
import torch.utils._pytree as pytree
import functools

from torchrl.data import CompositeSpec, TensorSpec, UnboundedContinuous
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, VecNorm, TensorDictPrimer
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule as TDMod,
    TensorDictSequential as TDSeq,
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union, Tuple
from collections import OrderedDict

from active_adaptation.learning.modules import IndependentNormal, VecNorm, GRUCore
from active_adaptation.learning.ppo.common import *
from active_adaptation.learning.ppo.ppo_base import PPOBase
from active_adaptation.utils.symmetry import SymmetryTransform

torch.set_float32_matmul_precision('high')

import active_adaptation
import torch.distributed as distr
from torch.nn.parallel import DistributedDataParallel as DDP

@dataclass
class PPOConfig:
    _target_: str = f"{__package__}.ppo_atec.PPOPolicy"
    name: str = "ppo_atec"
    train_every: int = 32
    ppo_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 5e-4
    desired_kl: Union[float, None] = None
    clip_param: float = 0.2
    entropy_coef: float = 0.002

    compile: bool = False
    phase: str = "train"
    symaug: bool = True

    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, OBS_PRIV_KEY)

cs = ConfigStore.instance()
cs.store("ppo_atec", node=PPOConfig, group="algo")
cs.store("ppo_atec_finetune", node=PPOConfig(phase="finetune", symaug=False), group="algo")


class GRUModule(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.mlp = make_mlp([128])
        self.gru = GRUCore(input_size=128, hidden_size=128)
        self.out = nn.LazyLinear(output_dim)
    
    def forward(self, x, is_init, hx):
        x = self.mlp(x)
        x, hx = self.gru(x, hx, is_init)
        x = self.out(x)
        return x, hx.contiguous()


class PPOPolicy(PPOBase):

    obs_keys = [CMD_KEY, OBS_KEY, OBS_PRIV_KEY]
    in_keys = [ACTION_KEY, "loc", "scale", "is_init", "action_log_prob", "hx"]
    in_keys = in_keys + [("next", obs_key) for obs_key in obs_keys]

    def __init__(
        self, 
        cfg: PPOConfig, 
        observation_spec: CompositeSpec, 
        action_spec: CompositeSpec, 
        reward_spec: TensorSpec,
        device,
        env=None,
    ):
        super().__init__()
        self.cfg = PPOConfig(**cfg)
        self.device = device

        self.observation_spec = observation_spec
        self.max_grad_norm = 1.0
        self.desired_kl = self.cfg.desired_kl
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.action_dim = env.action_manager.action_dim
        self.gae = GAE(0.99, 0.95)
        
        fake_input = observation_spec.zero()
        
        self.cmd_transform = env.observation_funcs[CMD_KEY].symmetry_transform().to(self.device)
        self.obs_transform = env.observation_funcs[OBS_KEY].symmetry_transform().to(self.device)
        self.priv_transform = env.observation_funcs[OBS_PRIV_KEY].symmetry_transform().to(self.device)
        self.act_transform = env.action_manager.symmetry_transform().to(self.device)
        self.input_transform = SymmetryTransform.cat([self.cmd_transform, self.obs_transform]).to(self.device)
        
        obs_cmd_dim = observation_spec[OBS_KEY].shape[-1] + observation_spec[CMD_KEY].shape[-1]
        priv_dim = observation_spec[OBS_PRIV_KEY].shape[-1]
        self.vecnorm = TDSeq(
            CatTensors([CMD_KEY, OBS_KEY], "_cmd_policy", del_keys=False, sort=False),
            TDMod(VecNorm(obs_cmd_dim, obs_cmd_dim, decay=1.0), ["_cmd_policy"], ["_cmd_policy"]),
            TDMod(VecNorm(priv_dim, priv_dim, decay=1.0), [OBS_PRIV_KEY], ["_priv"]),
        ).to(self.device)

        def make_actor(in_keys) -> ProbabilisticActor:
            return ProbabilisticActor(
                module=TDSeq(
                    CatTensors(in_keys, "_actor_input", del_keys=False, sort=False),
                    TDMod(make_mlp([256, 256, 256]), ["_actor_input"], ["_actor_input"]),
                    TDMod(Actor(self.action_dim), ["_actor_input"], ["loc", "scale"]),
                ),
                in_keys=["loc", "scale"],
                out_keys=[ACTION_KEY],
                distribution_class=IndependentNormal,
                return_log_prob=True
            ).to(self.device)

        self.priv_encoder = TDMod(
            nn.Sequential(make_mlp([128]), nn.LazyLinear(128)),
            ["_priv"], ["_priv_feature"]
        ).to(self.device)
        self.actor_teacher = make_actor(["_cmd_policy", "_priv_feature"])
        self.actor_student = make_actor(["_cmd_policy", "priv_feature_est"])
        
        self.adapt_module = TDMod(
            GRUModule(output_dim=128),
            ["_cmd_policy", "is_init", "hx"],
            ["priv_feature_est", ("next", "hx")]
        ).to(self.device)
        
        self.critic = TDSeq(
            CatTensors(["_cmd_policy", "_priv"], "_critic_input", del_keys=False, sort=False),
            TDMod(make_mlp([256, 256, 256]), ["_critic_input"], ["_critic_feature"]),
            TDMod(nn.LazyLinear(1), ["_critic_feature"], ["state_value"])
        ).to(self.device)

        with torch.device(self.device):
            fake_input["is_init"] = torch.ones(fake_input.shape[0], 1, dtype=torch.bool)
            fake_input["hx"] = torch.zeros(fake_input.shape[0], 128)
        
        self.vecnorm(fake_input)
        self.priv_encoder(fake_input)
        self.actor_teacher(fake_input)
        self.adapt_module(fake_input)
        self.actor_student(fake_input)
        self.critic(fake_input)

        self.opt_teacher = torch.optim.AdamW(
            [
                {"params": self.priv_encoder.parameters()},
                {"params": self.actor_teacher.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr,
            weight_decay=0.02
        )
        self.opt_student = torch.optim.AdamW(
            [
                {"params": self.actor_student.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr,
            weight_decay=0.02
        )

        self.opt_adapt = torch.optim.AdamW(
            [
                {"params": self.adapt_module.parameters()},
            ],
            lr=cfg.lr,
            weight_decay=0.02
        )
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor_teacher.apply(init_)
        self.critic.apply(init_)

        if active_adaptation.is_distributed():
            distr.init_process_group(
                backend="nccl",
                world_size=active_adaptation.get_world_size(),
                rank=active_adaptation.get_local_rank()
            )
            self.actor_teacher = DDP(self.actor_teacher)
            self.actor_student = DDP(self.actor_student)
            self.critic = DDP(self.critic)
            self.world_size = active_adaptation.get_world_size()
        
        self.update_teacher = functools.partial(
            self._update, 
            actor=TDSeq(self.priv_encoder, self.actor_teacher),
            opt=self.opt_teacher
        )
        self.update_student = functools.partial(
            self._update, 
            actor=self.actor_student,
            opt=self.opt_student
        )
    
    def make_tensordict_primer(self):
        num_envs = self.observation_spec.shape[0]
        return TensorDictPrimer(
            {"hx": UnboundedContinuous((num_envs, 128), device=self.device)},
            reset_key="done",
            expand_specs=False
        )
    
    def get_rollout_policy(self, mode: str="train"):
        if mode == "deploy":
            vecnorm = self.vecnorm[:2]
        else:
            vecnorm = self.vecnorm
        if self.cfg.phase == "train":
            policy = TDSeq(vecnorm, self.priv_encoder, self.actor_teacher)
        else:
            policy = TDSeq(vecnorm, self.adapt_module, self.actor_student)
        return policy.select_out_keys(ACTION_KEY, "loc", "scale", "action_log_prob", ("next", "hx"))

    @VecNorm.freeze()
    def train_op(self, tensordict: TensorDict):
        assert VecNorm.FROZEN, "VecNorm must be frozen before training"
        tensordict = tensordict.exclude("stats")
        infos = {}
        if self.cfg.phase == "train":
            infos.update(self.train_policy(tensordict.copy()))
            if self.num_updates % 2 == 0:
                infos.update(self.train_adapt(tensordict.copy()))
        elif self.cfg.phase == "finetune":
            infos.update(self.train_policy(tensordict.copy()))
            infos.update(self.train_adapt(tensordict.copy()))
        
        if active_adaptation.is_distributed():
            self.vecnorm[1].module.synchronize(mode="broadcast")
            self.vecnorm[2].module.synchronize(mode="broadcast")
        
        self.num_updates += 1
        return infos
    
    def train_policy(self, tensordict: TensorDict):
        infos = []

        self.vecnorm(tensordict)
        self.vecnorm(tensordict["next"])
        self.compute_advantage(tensordict, self.critic, "adv", "ret")

        action = tensordict[ACTION_KEY].clone()
        adv_unnormalized = tensordict["adv"]
        log_probs_before = tensordict["action_log_prob"]
        tensordict["adv"] = normalize(tensordict["adv"], subtract_mean=True)

        if self.cfg.phase == "train":
            update_fn = self.update_teacher
            actor = TDSeq(self.priv_encoder, self.actor_teacher)
        else:
            update_fn = self.update_student
            actor = self.actor_student

        for epoch in range(self.cfg.ppo_epochs):
            batches = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batches:
                infos.append(update_fn(minibatch))
                
                # if self.desired_kl is not None: # adaptive learning rate
                #     kl = infos[-1]["actor/kl"]
                #     actor_lr = self.opt.param_groups[0]["lr"]
                #     if kl > self.desired_kl * 2.0:
                #         actor_lr = max(1e-5, actor_lr / 1.5)
                #     elif kl < self.desired_kl / 2.0 and kl > 0.0:
                #         actor_lr = min(1e-2, actor_lr * 1.5)
                #     self.opt.param_groups[0]["lr"] = actor_lr

        with torch.no_grad():
            tensordict_ = actor(tensordict.copy())
            dist = IndependentNormal(tensordict_["loc"], tensordict_["scale"])
            log_probs_after = dist.log_prob(action)
            pg_loss_after = log_probs_after.reshape_as(adv_unnormalized) * adv_unnormalized
            pg_loss_before = log_probs_before.reshape_as(adv_unnormalized) * adv_unnormalized

        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        # infos["actor/lr"] = self.opt.param_groups[0]["lr"]
        infos["actor/pg_loss_raw_after"] = pg_loss_after.mean().item()
        infos["actor/pg_loss_raw_before"] = pg_loss_before.mean().item()
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_var"] = tensordict["ret"].var().item()
        infos["critic/neg_rew_ratio"] = (tensordict[REWARD_KEY].sum(-1) <= 0.).float().mean().item()
        return dict(sorted(infos.items()))

    def _update(self, tensordict: TensorDict, actor: TDMod, opt: torch.optim.Optimizer):
        bsize = tensordict.shape[0]
        loc_old, scale_old = tensordict["loc"], tensordict["scale"]

        if self.cfg.symaug:
            symmetry = tensordict.empty()
            symmetry[ACTION_KEY] = self.act_transform(tensordict[ACTION_KEY])
            symmetry["_priv"] = self.priv_transform(tensordict["_priv"])
            symmetry["_cmd_policy"] = self.input_transform(tensordict["_cmd_policy"])
            symmetry["action_log_prob"] = tensordict["action_log_prob"]
            symmetry["adv"] = tensordict["adv"]
            symmetry["ret"] = tensordict["ret"]
            symmetry["is_init"] = tensordict["is_init"]
            tensordict = torch.cat([tensordict.select(*symmetry.keys(True, True)), symmetry], dim=0)

        valid = (~tensordict["is_init"])
        valid_cnt = valid.sum()
        action_data = tensordict[ACTION_KEY]
        log_probs_data = tensordict["action_log_prob"]
        
        tensordict = actor(tensordict)
        tensordict = self.critic(tensordict)

        dist = IndependentNormal(tensordict["loc"], tensordict["scale"])
        log_probs = dist.log_prob(action_data)
        entropy = (dist.entropy().reshape_as(valid) * valid).sum() / valid_cnt

        adv = tensordict["adv"]
        log_ratio = (log_probs - log_probs_data).unsqueeze(-1)
        ratio = torch.exp(log_ratio)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - (torch.min(surr1, surr2).reshape_as(valid) * valid).sum() / valid_cnt
        entropy_loss = - self.cfg.entropy_coef * entropy

        b_returns = tensordict["ret"]
        values = tensordict["state_value"]
        value_loss = self.critic_loss_fn(b_returns, values)
        value_loss = (value_loss.reshape_as(valid) * valid).sum() / valid_cnt

        loss = policy_loss + entropy_loss + value_loss
        opt.zero_grad()
        loss.backward()

        actor_grad_norm = nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
        critic_grad_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        opt.step()
        
        info = {
            "actor/policy_loss": policy_loss.detach(),
            "actor/entropy": entropy.detach(),
            "actor/grad_norm": actor_grad_norm,
            "critic/value_loss": value_loss.detach(),
            "critic/grad_norm": critic_grad_norm,
        }
        with torch.no_grad():
            info["critic/explained_var"] = 1 - value_loss / b_returns[valid].var()
            info["actor/clamp_ratio"] = ((ratio - 1.0).abs() > self.clip_param).float().mean()
            loc, scale = dist.loc[:bsize], dist.scale[:bsize]
            info["actor/kl"] = IndependentNormal.kl(loc, scale, loc_old, scale_old).mean()
            if self.cfg.symaug:
                symmetry_loss = F.mse_loss(dist.mean[bsize:], self.act_transform(dist.mean[:bsize]))
                info["actor/symmetry_loss"] = symmetry_loss
        return info
    
    def train_adapt(self, tensordict: TensorDict):
        with torch.inference_mode():
            self.vecnorm(tensordict)
            self.priv_encoder(tensordict)

        batches = make_batch(tensordict, self.cfg.num_minibatches, self.cfg.train_every)
        for minibatch in batches:
            self.adapt_module(minibatch)
            valid = (~minibatch["is_init"])
            valid_cnt = valid.sum()
            adapt_loss = torch.square(minibatch["priv_feature_est"] - minibatch["_priv_feature"]).mean(-1, keepdim=True)
            adapt_loss = (adapt_loss * valid).sum() / valid_cnt
            self.opt_adapt.zero_grad()
            adapt_loss.backward()
            self.opt_adapt.step()
        return {
            "adapt/priv_loss": adapt_loss.detach().item(),
        }
    
    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["last_phase"] = self.cfg.phase
        return state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        failed_keys = super().load_state_dict(state_dict, strict=strict)
        if state_dict.get("last_phase", "train") == "train":
            hard_copy_(self.actor_teacher, self.actor_student)
        if active_adaptation.is_distributed():
            self.vecnorm[1].module.synchronize(mode="broadcast")
            self.vecnorm[2].module.synchronize(mode="broadcast")
        return failed_keys


def normalize(x: torch.Tensor, subtract_mean: bool=False):
    if subtract_mean:
        return (x - x.mean()) / x.std().clamp(1e-7)
    else:
        return x  / x.std().clamp(1e-7)
