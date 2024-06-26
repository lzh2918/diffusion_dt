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
import re

from torch.utils.tensorboard import SummaryWriter

# uper vf
from value_func.uper_value_function import Value_function_Transformer
from value_func.dt_sequence import SequenceHalfcondTimestepDataset
from config.locomotion_config import Config

from batch_dt_upervf_under import train


def extract_numbers_after_char(string, char):
    pattern = f'{char}(\\d+)'
    matches = re.findall(pattern, string)
    return matches


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
    env_name: str = "walker2d-medium-expert-v2" # 这里要求训练diffusion的数据集，和训练dt的数据集是同一个数据集。
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    clip_grad: Optional[float] = 0.25
    batch_size: int = 64
    update_steps: int = 150_000
    warmup_steps: int = 10_000
    reward_scale: float = 0.001
    num_workers: int = 4
    # evaluation params
    target_returns: str = "(2000.0_2500.0_3000.0)"
    eval_episodes: int = 20
    eval_every: int = 50_00 # 调试修改过
    # eval_every: int = 2000 # 调试修改过
    # general params
    checkpoints_path: Optional[str] = None
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    device: str = "cuda"
    # new add 
    horizon: int = 20
    generate_percentage_list: str = "(0.0_0.2_0.6_1.0)"
    generate_percentage: float = 0.5
    diffusion_data_load_path: str = "/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion_store_data/longtime_upervf/walker2d-medium-expert-v2/diff_horizon_20_cond_length_2/uper_vf_er_0.95cond_length2_layer_3_head_1/diff_date_24-0608-210615_upervf_date_24-0618-143752/24-0620-140439/save_traj.npy"
    return_change_coef: float = 1.0
    dataset_scale: str = "(1.0_2.0)"
    # save
    save_model: bool = True
    # uper value func
    cond_length: int = 2
    discount: float = 1.0
    uper_vf_path: str = "/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/walker2d-medium-expert-v2/er_0.95cond_length2_layer_3_head_1/24-0618-143752/uper_value_func_checkpoint.pt"
    # eval 
    eval_batch: int = 10

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

@pyrallis.wrap()
def main(config: TrainConfig):
    circle_time=3
    for i in range(circle_time):
        trian_seed = np.random.randint(0,100,1)[0]
        eval_seed = np.random.randint(0,100,1)[0]
        config.train_seed = trian_seed
        config.eval_seed = eval_seed

        # 设置upervf参数
        upervf_tag = config.uper_vf_path.split('/')[-3]
        num_layers = int(extract_numbers_after_char(upervf_tag,"layer_")[0])
        num_heads = int(extract_numbers_after_char(upervf_tag,"head_")[0])
        config.num_layers = num_layers
        config.num_heads = num_heads

        # 下面的值就相当于给一个下界，不能给的太高。
        if "halfcheetah" in config.env_name:
            config.target_returns = "(0.0_12000.0_9000.0)"
        elif "hopper" in config.env_name:
            config.target_returns = "(0.0_3000.0_6000.0)"
        elif "walker2d" in config.env_name:
            config.target_returns = "(6000.0_3500.0_0.0)"
        assert config.env_name in config.uper_vf_path
        assert config.env_name in config.diffusion_data_load_path
        generate_percentage_list = ast.literal_eval(config.generate_percentage_list.replace("_",", "))
        if type(generate_percentage_list) == float:
            generate_percentage_list = [generate_percentage_list]
        for generate_percentage in generate_percentage_list:
            config.generate_percentage = generate_percentage
            train(config)

if __name__ == "__main__":
    main()

