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
import warnings
import functools
import torch.utils._pytree as pytree

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors, VecNorm
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as TDMod,
    TensorDictSequential,
)

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union, Tuple
from collections import OrderedDict

from active_adaptation.learning.utils.valuenorm import ValueNorm1, ValueNormFake
from active_adaptation.learning.modules.distributions import IndependentNormal
from active_adaptation.learning.ppo.common import *

torch.set_float32_matmul_precision('high')
USE_DDP = True

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
    entropy_coef: float = 0.01
    layer_norm: Union[str, None] = "before"
    value_norm: bool = False
    compile: bool = False

    checkpoint_path: Union[str, None] = None
    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, "terrain_")

cs = ConfigStore.instance()
cs.store("ppo_atec", node=PPOConfig, group="algo")

class MixedEncoder(nn.Module):
    def __init__(self, mlp_out=256, cnn_out=32, conv3d: bool=False):
        super().__init__()
        self.mlp_out = mlp_out
        self.cnn_out = cnn_out
        self.mlp_encoder = make_mlp([256, 256])

        cnn_cls = nn.LazyConv2d
        data_dim = 3 # [C, X, Y]

        self.cnn_encoder = nn.Sequential(
            FlattenBatch(
                nn.Sequential(
                    cnn_cls(8, kernel_size=3, stride=2, padding=1), 
                    nn.Mish(), # nn.GroupNorm(num_channels=2, num_groups=2),
                    cnn_cls(8, kernel_size=3, stride=2, padding=1),
                    nn.Mish(), # nn.GroupNorm(num_channels=4, num_groups=2),
                    cnn_cls(8, kernel_size=3, stride=2, padding=1),
                    nn.Mish(), # nn.GroupNorm(num_channels=8, num_groups=2), 
                    nn.Flatten(),
                ),
                data_dim=data_dim,
            ),
            nn.LazyLinear(48),
            nn.Mish(),
            nn.LayerNorm(48),
        )
        self.out = nn.Sequential(nn.LazyLinear(256), nn.Mish())

    def forward(self, mlp_inp, cnn_inp, mask_cnn=None):
        # cnn_feature = self.cnn_encoder(cnn_inp.float())
        mlp_feature = self.mlp_encoder(mlp_inp)
        # if mask_cnn is not None:
        #     cnn_feature = cnn_feature * mask_cnn
        # feature = torch.cat([mlp_feature, cnn_feature], dim=-1)
        return self.out(mlp_feature)


class PPOPolicy(TensorDictModuleBase):

    actor_in_keys = [CMD_KEY, OBS_KEY]
    critic_in_keys = [CMD_KEY, OBS_KEY]

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

        self.entropy_coef = self.cfg.entropy_coef
        self.max_grad_norm = 1.0
        self.desired_kl = self.cfg.desired_kl
        self.clip_param = self.cfg.clip_param
        self.critic_loss_fn = nn.MSELoss(reduction="none")
        self.action_dim = action_spec.shape[-1]
        self.gae = GAE(0.99, 0.95)
        
        if cfg.value_norm:
            value_norm_cls = ValueNorm1
        else:
            value_norm_cls = ValueNormFake
        self.value_norm = value_norm_cls(input_shape=1).to(self.device)

        fake_input = observation_spec.zero()
        
        self.cmd_transform = env.observation_funcs[CMD_KEY].symmetry_transforms()
        self.obs_transform = env.observation_funcs[OBS_KEY].symmetry_transforms()
        self.terrain_transform = env.observation_funcs["terrain_"].symmetry_transforms()
        self.act_transform = env.action_manager.symmetry_transforms()
        self.cmd_transform = self.cmd_transform.to(self.device)
        self.obs_transform = self.obs_transform.to(self.device)
        self.terrain_transform = self.terrain_transform.to(self.device)
        self.act_transform = self.act_transform.to(self.device)

        actor_module = TensorDictSequential(
            CatTensors(self.actor_in_keys, "_actor_input", del_keys=False, sort=False),
            TDMod(MixedEncoder(conv3d=False), ["_actor_input", "terrain_"], ["_actor_feature"]),
            TDMod(Actor(self.action_dim), ["_actor_feature"], ["loc", "scale"])
        )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[ACTION_KEY],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)
        
        self.critic = TensorDictSequential(
            CatTensors(self.critic_in_keys, "_critic_input", del_keys=False, sort=False),
            TDMod(MixedEncoder(conv3d=False), ["_critic_input", "terrain_"], ["_critic_feature"]),
            TDMod(nn.LazyLinear(1), ["_critic_feature"], ["state_value"])
        ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)

        self.opt = torch.optim.AdamW(
            [
                {"params": self.actor.parameters()},
                {"params": self.critic.parameters()},
            ],
            lr=cfg.lr,
            weight_decay=0.02
        )
        
        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)
        
        self.actor.apply(init_)
        self.critic.apply(init_)

        if active_adaptation.is_distributed():
            distr.init_process_group(
                backend="nccl",
                world_size=active_adaptation.get_world_size(),
                rank=active_adaptation.get_local_rank()
            )
            if USE_DDP:
                self.actor = DDP(self.actor)
                self.critic = DDP(self.critic)
            else:
                for param in self.actor.parameters():
                    distr.broadcast(param, src=0)
                for param in self.critic.parameters():
                    distr.broadcast(param, src=0)
            self.world_size = active_adaptation.get_world_size()
            
        self.update = self._update
        if self.cfg.compile and not active_adaptation.is_distributed():
            # TODO: compile for multi-gpu training?
            self.update = torch.compile(self.update, fullgraph=True)
            # self.update = CudaGraphModule(self.update)
    
    def get_rollout_policy(self, mode: str="train"):
        policy = TensorDictSequential(self.actor)
        if self.cfg.compile:
            policy = torch.compile(policy)
            # policy = CudaGraphModule(policy)
        return policy

    def train_op(self, tensordict: TensorDict):
        tensordict = tensordict.exclude("stats")
        infos = []
        self._compute_advantage(tensordict, self.critic, "adv", "ret", update_value_norm=True)
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
                
        infos = pytree.tree_map(lambda *xs: sum(xs).item() / len(xs), *infos)
        infos["actor/lr"] = self.opt.param_groups[0]["lr"]
        infos["critic/value_mean"] = tensordict["ret"].mean().item()
        infos["critic/value_var"] = tensordict["ret"].var().item()
        infos["critic/neg_rew_ratio"] = (tensordict[REWARD_KEY].sum(-1) <= 0.).float().mean().item()
        return dict(sorted(infos.items()))

    @torch.no_grad()
    def _compute_advantage(
        self, 
        tensordict: TensorDict,
        critic: TDMod, 
        adv_key: str="adv",
        ret_key: str="ret",
        update_value_norm: bool=True,
    ):
        keys = tensordict.keys(True, True)
        if not ("state_value" in keys and ("next", "state_value") in keys):
            with tensordict.view(-1) as tensordict_flat:
                critic(tensordict_flat)
                critic(tensordict_flat["next"])

        values = tensordict["state_value"]
        next_values = tensordict["next", "state_value"]

        rewards = tensordict[REWARD_KEY].sum(-1, keepdim=True).clamp_min(0.)
        discount = tensordict["next", "discount"]
        terms = tensordict[TERM_KEY]
        dones = tensordict[DONE_KEY]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, terms, dones, values, next_values, discount)
        if update_value_norm:
            self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set(adv_key, adv)
        tensordict.set(ret_key, ret)
        return tensordict

    def _update(self, tensordict: TensorDict):
        bsize = tensordict.shape[0]
        loc_old, scale_old = tensordict["loc"], tensordict["scale"]

        symmetry = tensordict.empty()
        symmetry[ACTION_KEY] = self.act_transform(tensordict[ACTION_KEY])
        symmetry[CMD_KEY] = self.cmd_transform(tensordict[CMD_KEY])
        symmetry[OBS_KEY] = self.obs_transform(tensordict[OBS_KEY])
        symmetry["terrain_"] = self.terrain_transform(tensordict["terrain_"])
        symmetry["action_log_prob"] = tensordict["action_log_prob"]
        symmetry["adv"] = tensordict["adv"]
        symmetry["ret"] = tensordict["ret"]
        symmetry["is_init"] = tensordict["is_init"]
        tensordict = torch.cat([tensordict.select(*symmetry.keys(True, True)), symmetry], dim=0)

        valid = (~tensordict["is_init"])
        valid_cnt = valid.sum()
        action_data = tensordict[ACTION_KEY]
        log_probs_data = tensordict["action_log_prob"]
        self.actor(tensordict)
        dist = IndependentNormal(tensordict["loc"], tensordict["scale"])
        # dist: IndependentNormal = self.actor.get_dist(tensordict)
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

        if active_adaptation.is_distributed() and not USE_DDP:
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
            kl = torch.sum(
                torch.log(scale) - torch.log(scale_old)
                + (torch.square(scale_old) + torch.square(loc_old - loc)) / (2.0 * torch.square(scale))
                - 0.5,
                axis=-1,
            ).mean()
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
