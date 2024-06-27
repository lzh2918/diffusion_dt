# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import os
current_upup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(current_upup_dir)

import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import d4rl
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

import datetime
import ast
from iqlvf import train

TensorBatch = List[torch.Tensor]


EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

"""
需要做的事情 1.wandb改成tensorboard  2.eval 可能要加快一点  3.加上一些其他的统计量
4. 要加上真实return 的那一项，就需要改动数据流，需要一点时间。
"""


@dataclass
class TrainConfig:
    # Wandb logging
    project: str = "upervf"
    group: str = "IQLtest"
    name: str = "IQLvf"
    # Experiment
    device: str = "cuda"
    env: str = "hopper-medium-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    # eval_freq: int = 10  # 调试用
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    # iql_tau: float = 0.7  # Coefficient for asymmetric loss
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # loss
    real_weight: float = 0.25
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    # new add
    reward_scale: float = float(1/400)
    save_checkpoints: bool = True
    discount_list: str = "(1.0_0.99)"
   

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


@pyrallis.wrap()
def main(config: TrainConfig):
    discount_list = ast.literal_eval(config.discount_list.replace("_",", "))
    if type(discount_list) == float:
        discount_list = [discount_list]
    for discount in discount_list:
        config.discount = discount
        train(config)

if __name__ == "__main__":
    main()
