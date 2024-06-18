# inspiration:
# 1. https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py  # noqa
# 2. https://github.com/karpathy/minGPT
import os
current_upup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(current_upup_dir)

import os
import random
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

import d4rl  # noqa
import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import wandb
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm, trange  # noqa

from my_add_code.traj_buffer import traj_buffer
import datetime
import ast

from torch.utils.tensorboard import SummaryWriter

# uper vf
from value_func.uper_value_function import Value_function_Transformer
from value_func.dt_sequence import SequenceHalfcondTimestepDataset
from config.locomotion_config import Config

from batch_dt_upervf_under import train

circle_time=3

@dataclass
class TrainConfig:
    # wandb params
    project: str = "dt"
    group: str = "test"
    name: str = "the original"
    # model params
    embedding_dim: int = 128
    num_layers: int = 3
    num_heads: int = 1
    seq_len: int = 20
    episode_len: int = 1000
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    max_action: float = 1.0
    # training params
    env_name: str = "hopper-medium-v2" # 这里要求训练diffusion的数据集，和训练dt的数据集是同一个数据集。
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 128
    update_steps: int = 150_000
    warmup_steps: int = 10_000
    reward_scale: float = 0.001
    num_workers: int = 4
    # evaluation params
    target_returns: str = "(2000.0_2500.0_3000.0)"
    eval_episodes: int = 20
    eval_every: int = 10_00 # 调试修改过
    # eval_every: int = 10 # 调试修改过
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    device: str = "cuda"
    # new add 
    horizon: int = 20
    generate_percentage: float = 0.5
    diffusion_data_load_path: str = "/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/store_data/hopper-medium-v2/diffusion_horizon_20_cond_length_1024-0606-225731/24-0610-214516_er_0.95_cond_length_10/save_traj.npy"
    return_change_coef: float = 1.0
    dataset_scale: str = "(1.0_2.0)"
    # save
    save_model: bool = True
    # uper value func
    cond_length: int = 10
    discount: float = 1.0
    uper_vf_path: str = "/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-v2/er_0.95_cond_length_10/24-0611-161800/uper_value_func_checkpoint.pt"
    # eval 
    eval_batch: int = 10

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

for i in range(circle_time):
    trian_seed = np.random.randint(0,100,1)[0]
    eval_seed = np.random.randint(0,100,1)[0]
    config = TrainConfig()
    config.train_seed = trian_seed
    config.eval_seed = eval_seed
    train(config)