# inspiration:
# 1. https://github.com/kzl/decision-transformer/blob/master/gym/decision_transformer/models/decision_transformer.py  # noqa
# 2. https://github.com/karpathy/minGPT
"""
这是6月17号进行修改的版本。最新版本。
"""
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
from diffuser.datasets.sequence import SequenceHalfcondTimestepDataset
import diffuser.utils as utils
import datetime
import ast
from config.locomotion_config import Config

from torch.utils.tensorboard import SummaryWriter

# dataset 
from value_func.dt_sequence import SequenceHalfcondTimestepDataset


@dataclass
class TrainConfig:
    # wandb params
    project: str = "diffusion_dt"
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
    update_steps: int = 200_0000
    warmup_steps: int = 10_000
    reward_scale: float = 0.001
    num_workers: int = 4
    # evaluation params
    target_returns: str = "(2000.0_2500.0_3000.0)"
    eval_episodes: int = 10
    eval_every: int = 10_00
    # general params
    checkpoints_path: Optional[str] = None 
    deterministic_torch: bool = False
    train_seed: int = 10
    eval_seed: int = 42
    device: str = "cuda"
    # new add 
    horizon: int = 20
    generate_percentage: float = 0.5
    # diffusion_data_load_path: str = "/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser_collect_data/half_cond_1/hopper-medium-v2/horizon_20/24-0527-103632/hopper-medium-v2/24-0528-2052391.0_2.0/save_traj.npy"
    return_change_coef: float = 1.0
    dataset_scale: str = "(1.0_2.0)"
    cond_length: int = 10
    # dataload
    data_loader: str = "value_func.SequenceHalfcondTimestepDataset"
    discount: float = 1.0
    # expect_regression
    er_coef: float = 0.5
    # save
    root_save_path: str = "/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/collect_data"
    save_checkpoints: bool = False



    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


# general utils
def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    name = "hor_" + str(config["horizon"]) + "_geper_" + str(config["generate_percentage"]) + "_data_scale_" + f"[{config['dataset_scale'][0]},{config['dataset_scale'][1]}]_"
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=name+timestamp,
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


# some utils functionalities specific for Decision Transformer
def pad_along_axis(
    arr: np.ndarray, pad_to: int, axis: int = 0, fill_value: float = 0.0
) -> np.ndarray:
    pad_size = pad_to - arr.shape[axis]
    if pad_size <= 0:
        return arr

    npad = [(0, 0)] * arr.ndim
    npad[axis] = (0, pad_size)
    return np.pad(arr, pad_width=npad, mode="constant", constant_values=fill_value)


def discounted_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum


def load_d4rl_trajectories(
    env_name: str, gamma: float = 1.0
) -> Tuple[List[DefaultDict[str, np.ndarray]], Dict[str, Any]]:
    dataset = gym.make(env_name).get_dataset()
    traj, traj_len = [], []

    data_ = defaultdict(list)
    for i in trange(dataset["rewards"].shape[0], desc="Processing trajectories"):
        data_["observations"].append(dataset["observations"][i])
        data_["actions"].append(dataset["actions"][i])
        data_["rewards"].append(dataset["rewards"][i])

        if dataset["terminals"][i] or dataset["timeouts"][i]:
            episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
            # return-to-go if gamma=1.0, just discounted returns else
            episode_data["returns"] = discounted_cumsum(
                episode_data["rewards"], gamma=gamma
            )
            traj.append(episode_data)
            traj_len.append(episode_data["actions"].shape[0])
            # reset trajectory buffer
            data_ = defaultdict(list)

    # needed for normalization, weighted sampling, other stats can be added also
    info = {
        "obs_mean": dataset["observations"].mean(0, keepdims=True),
        "obs_std": dataset["observations"].std(0, keepdims=True) + 1e-6,
        "traj_lens": np.array(traj_len),
    }
    return traj, info


class SequenceDataset(IterableDataset):
    def __init__(self, env_name: str, seq_len: int = 10, reward_scale: float = 1.0):
        self.dataset, info = load_d4rl_trajectories(env_name, gamma=1.0)
        self.reward_scale = reward_scale
        self.seq_len = seq_len

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["observations"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        returns = traj["returns"][start_idx : start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        # pad up to seq_len if needed
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        return states, actions, returns, time_steps, mask

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
            yield self.__prepare_sample(traj_idx, start_idx)


class MixSequenceDataset(IterableDataset):
    def __init__(self, env_name: str, traj_buffer, seq_len: int = 10, reward_scale: float = 1.0, gen_per: float = 0.5):
        self.dataset, info = load_d4rl_trajectories(env_name, gamma=1.0)
        self.gen_per = gen_per
        self.traj_buffer = traj_buffer
        self.reward_scale = reward_scale
        self.seq_len = seq_len

        self.state_mean = info["obs_mean"]
        self.state_std = info["obs_std"]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L116 # noqa
        self.sample_prob = info["traj_lens"] / info["traj_lens"].sum()

    def __prepare_sample(self, traj_idx, start_idx):
        traj = self.dataset[traj_idx]
        # https://github.com/kzl/decision-transformer/blob/e2d82e68f330c00f763507b3b01d774740bee53f/gym/experiment.py#L128 # noqa
        states = traj["observations"][start_idx : start_idx + self.seq_len]
        actions = traj["actions"][start_idx : start_idx + self.seq_len]
        returns = traj["returns"][start_idx : start_idx + self.seq_len]
        time_steps = np.arange(start_idx, start_idx + self.seq_len)

        states = (states - self.state_mean) / self.state_std
        returns = returns * self.reward_scale
        # pad up to seq_len if needed
        mask = np.hstack(
            [np.ones(states.shape[0]), np.zeros(self.seq_len - states.shape[0])]
        )
        if states.shape[0] < self.seq_len:
            states = pad_along_axis(states, pad_to=self.seq_len)
            actions = pad_along_axis(actions, pad_to=self.seq_len)
            returns = pad_along_axis(returns, pad_to=self.seq_len)

        return states, actions, returns, time_steps, mask

    def __iter__(self):
        while True:
            traj_idx = np.random.choice(len(self.dataset), p=self.sample_prob)
            start_idx = random.randint(0, self.dataset[traj_idx]["rewards"].shape[0] - 1)
            if np.random.rand(1)[0] > self.gen_per:
                yield self.__prepare_sample(traj_idx, start_idx)
            else:
                yield self.traj_buffer.sample_one(self.reward_scale, self.state_mean, self.state_std)


# Decision Transformer implementation
class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len: int,
        embedding_dim: int,
        num_heads: int,
        attention_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        # True value indicates that the corresponding position is not allowed to attend
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]

        norm_x = self.norm1(x)
        attention_out = self.attention(
            query=norm_x,
            key=norm_x,
            value=norm_x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        # by default pytorch attention does not use dropout
        # after final attention weights projection, while minGPT does:
        # https://github.com/karpathy/minGPT/blob/7218bcfa527c65f164de791099de715b81a95106/mingpt/model.py#L70 # noqa
        x = x + self.drop(attention_out)
        x = x + self.mlp(self.norm2(x))
        return x


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        max_action: float = 1.0,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim)
        self.state_emb = nn.Linear(state_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.return_emb = nn.Linear(1, embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=3 * seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Sequential(nn.Linear(embedding_dim, action_dim), nn.Tanh())
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.episode_len = episode_len
        self.max_action = max_action

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,  # [batch_size, seq_len, state_dim]
        actions: torch.Tensor,  # [batch_size, seq_len, action_dim]
        returns_to_go: torch.Tensor,  # [batch_size, seq_len]
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = states.shape[0], states.shape[1]
        # [batch_size, seq_len, emb_dim]
        time_emb = self.timestep_emb(time_steps)
        state_emb = self.state_emb(states) + time_emb
        act_emb = self.action_emb(actions) + time_emb
        returns_emb = self.return_emb(returns_to_go.unsqueeze(-1)) + time_emb

        # [batch_size, seq_len * 3, emb_dim], (r_0, s_0, a_0, r_1, s_1, a_1, ...)
        sequence = (
            torch.stack([returns_emb, state_emb, act_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 3 * seq_len, self.embedding_dim)
        )
        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(sequence)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out)
        # [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        out = self.action_head(out[:, 1::3]) * self.max_action
        return out
    
class Value_function_Transformer(nn.Module):
    def __init__(
        self,
        transition_dim: int = 15, # state + action + reward
        seq_len: int = 10,
        episode_len: int = 1000,
        embedding_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        attention_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        # max_action: float = 1.0,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        self.out_norm = nn.LayerNorm(embedding_dim)
        # additional seq_len embeddings for padding timesteps
        #要不这个改成position embedding？关系大吗？
        self.timestep_emb = nn.Embedding(episode_len + seq_len, embedding_dim) # 这个time embedding 和 position embedding的关系怎么说？说不清楚啊，有点
        # self.state_emb = nn.Linear(state_dim, embedding_dim)
        # self.action_emb = nn.Linear(action_dim, embedding_dim)
        # self.return_emb = nn.Linear(1, embedding_dim)
        self.traj_embedder = nn.Sequential(
            nn.Linear(transition_dim, embedding_dim),
            nn.Mish(),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=seq_len, # 进行了修改
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )
        # self.action_head = nn.Sequential(nn.Linear(embedding_dim, action_dim), nn.Tanh())
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim*seq_len, embedding_dim),
            nn.Mish(),
            nn.Linear(embedding_dim, 1)
        )
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.transition_dim = transition_dim
        self.episode_len = episode_len
        # self.max_action = max_action

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        trajectories: torch.Tensor,
        time_steps: torch.Tensor,  # [batch_size, seq_len]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.FloatTensor:
        batch_size, seq_len = trajectories.shape[0], trajectories.shape[1]
        time_emb = self.timestep_emb(time_steps)
        traj_emb = self.traj_embedder(trajectories) + time_emb

        if padding_mask is not None:
            # [batch_size, seq_len * 3], stack mask identically to fit the sequence
            padding_mask = (
                torch.stack([padding_mask, padding_mask, padding_mask], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_len)
            )
        # LayerNorm and Dropout (!!!) as in original implementation,
        # while minGPT & huggingface uses only embedding dropout
        out = self.emb_norm(traj_emb)
        out = self.emb_drop(out)

        for block in self.blocks:
            out = block(out, padding_mask=padding_mask)

        out = self.out_norm(out).flatten(1,2)
        # [batch_size, seq_len, action_dim]
        # predict actions only from state embeddings
        # out = self.action_head(out[:, 1::3]) * self.max_action
        value = self.value_head(out)
        return value


# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    model: DecisionTransformer,
    env: gym.Env,
    target_return: float,
    device: str = "cpu",
) -> Tuple[float, float]:
    states = torch.zeros(
        1, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device
    )
    actions = torch.zeros(
        1, model.episode_len, model.action_dim, dtype=torch.float, device=device
    )
    returns = torch.zeros(1, model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(model.episode_len, dtype=torch.long, device=device)
    time_steps = time_steps.view(1, -1)

    states[:, 0] = torch.as_tensor(env.reset(), device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash
    episode_return, episode_len = 0.0, 0.0
    for step in range(model.episode_len):
        # first select history up to step, then select last seq_len states,
        # step + 1 as : operator is not inclusive, last action is dummy with zeros
        # (as model will predict last, actual last values are not important)
        predicted_actions = model(  # fix this noqa!!!
            states[:, : step + 1][:, -model.seq_len :],
            actions[:, : step + 1][:, -model.seq_len :],
            returns[:, : step + 1][:, -model.seq_len :],
            time_steps[:, : step + 1][:, -model.seq_len :],
        )
        predicted_action = predicted_actions[0, -1].cpu().numpy()
        next_state, reward, done, info = env.step(predicted_action)
        # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
        actions[:, step] = torch.as_tensor(predicted_action)
        states[:, step + 1] = torch.as_tensor(next_state)
        returns[:, step + 1] = torch.as_tensor(returns[:, step] - reward)

        episode_return += reward
        episode_len += 1

        if done:
            break

    return episode_return, episode_len


def cycle(dl):
    while True:
        for data in dl:
            yield data


@pyrallis.wrap()
def train(config: TrainConfig):
    torch.set_num_threads(8)
    config.target_returns = ast.literal_eval(config.target_returns.replace("_",", "))
    config.dataset_scale = ast.literal_eval(config.dataset_scale.replace("_",", "))
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    # init wandb session for logging
    root_dir = "/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/collect_data"
    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    name = "hor_" + str(config.horizon) + "_er_coef_" + str(config.er_coef) + "_cond_length_" + f"{config.cond_length}_" + timestamp
    log_path = os.path.join(root_dir, config.project+"_tb", config.group, config.env_name, name)
    writer = SummaryWriter(log_dir=log_path)

    dataset = SequenceHalfcondTimestepDataset(
        env=config.env_name,
        horizon=config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
        discount=config.discount,
        cond_length=config.cond_length,
    )

    config.state_dim = dataset.observation_dim
    config.action_dim = dataset.action_dim
    # model & optimizer & scheduler setup
    model = Value_function_Transformer(
        transition_dim=config.state_dim + config.action_dim, # action + state no reward
        embedding_dim=config.embedding_dim,
        seq_len=config.cond_length,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        # max_action=config.max_action,
    ).to(config.device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda steps: min((steps + 1) / config.warmup_steps, 1),
    )
    # save config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    condition_data_loader = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))

    for step in trange(config.update_steps, desc="Training"):
        batch = next(condition_data_loader)
        batch = utils.arrays.batch_to_device(batch, device=config.device)
        returns = batch.returns
        time_steps = batch.timesteps
        trajectories = batch.conditions[0]

        predicted_returns = model(
            trajectories=trajectories,
            time_steps=time_steps,
        )

        delta = (predicted_returns.detach() - returns) > 0
        loss = (torch.abs(config.er_coef - delta.int()) * (returns - predicted_returns) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        total_length = len(predicted_returns)
        np_predicted_returns = predicted_returns.detach().cpu().numpy()
        np_returns = returns.cpu().numpy()
        sorted_predicted_returns = np.sort(np_predicted_returns,axis=0)
        predicted_returns_sort_index = np.argsort(np_predicted_returns,axis=0).astype(np.int32)
        # inverse_predicted_returns_map = np.zeros_like(predicted_returns_sort_index)
        # for i in range(len(np_predicted_returns)):
        #     inverse_predicted_returns_map[predicted_returns_sort_index[i]] = i
        sorted_returns = np.sort(np_returns,axis=0)
        returns_sort_index = np.argsort(np_returns,axis=0).astype(np.int32)
        # inverse_returns_map = np.zeros_like(returns_sort_index)
        # for i in range(len(np_returns)):
        #     inverse_returns_map[returns_sort_index[i]] = i

        writer.add_scalar("train/train_loss", loss.item(), step)
        writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], step)
        if step % 1000 == 0:
            writer.add_scalar("eval/predict_return", predicted_returns.mean(), step)
            writer.add_scalar("eval/returns", np_returns.mean(), step)
            writer.add_scalar("eval/predict_return_last10", sorted_predicted_returns[:total_length//10].mean(), step)
            writer.add_scalar("eval/predict_return_top10", sorted_predicted_returns[-total_length//10:].mean(), step)
            writer.add_scalar("eval/return_last10", sorted_returns[:total_length//10].mean(), step)
            writer.add_scalar("eval/return_top10", sorted_returns[-total_length//10:].mean(), step)
            writer.add_scalar("eval/predict_max", sorted_predicted_returns[-1].mean(), step)
            writer.add_scalar("eval/predict_min", sorted_predicted_returns[0].mean(), step)
            writer.add_scalar("eval/real_return_max", sorted_returns[-1].mean(), step)
            writer.add_scalar("eval/real_return_min", sorted_returns[0].mean(), step)
            writer.add_scalar("eval/predict_top10_error_mean", (sorted_predicted_returns[-total_length//10:]-np.squeeze(np_returns[predicted_returns_sort_index [-total_length//10:]],axis=-1)).mean(), step)
            writer.add_scalar("eval/predict_top10_error_abs_mean", abs((sorted_predicted_returns[-total_length//10:]-np.squeeze(np_returns[predicted_returns_sort_index [-total_length//10:]],axis=-1))).mean(), step)
            writer.add_scalar("eval/predict_last10_error_mean", (sorted_predicted_returns[:total_length//10]-np.squeeze(np_returns[predicted_returns_sort_index [:total_length//10]],axis=-1)).mean(), step)
            writer.add_scalar("eval/predict_last10_error_abs_mean", abs((sorted_predicted_returns[:total_length//10]-np.squeeze(np_returns[predicted_returns_sort_index [:total_length//10]],axis=-1))).mean(), step)
            writer.add_scalar("eval/return_t10-pre_return", (sorted_returns[-total_length//10:]-np.squeeze(np_predicted_returns[returns_sort_index[-total_length//10:]])).mean(), step)
            writer.add_scalar("eval/return_t10-pre_return_abs", abs(sorted_returns[-total_length//10:]-np.squeeze(np_predicted_returns[returns_sort_index[-total_length//10:]])).mean(), step)
            writer.add_scalar("eval/return_last10-pre_return", (sorted_returns[:total_length//10]-np.squeeze(np_predicted_returns[returns_sort_index[:total_length//10]])).mean(), step)
            writer.add_scalar("eval/return_last10-pre_return_abs", abs(sorted_returns[:total_length//10]-np.squeeze(np_predicted_returns[returns_sort_index[:total_length//10]])).mean(), step)
            if step == int(config.update_steps*0.7):
                if config.save_checkpoints is not None:
                    checkpoint = {
                        "model_state": model.state_dict(),
                        # "state_mean": dataset.state_mean,
                        # "state_std": dataset.state_std,
                    }
                    save_path = os.path.join(config.root_save_path, config.project, config.group, config.env_name, f"er_{config.er_coef}_cond_length_{config.cond_length}",timestamp)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(checkpoint, os.path.join(save_path, "uper_value_func_checkpoint_0.7.pt"))

    if config.save_checkpoints is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            # "state_mean": dataset.state_mean,
            # "state_std": dataset.state_std,
        }
        save_path = os.path.join(config.root_save_path, config.project, config.group, config.env_name, f"er_{config.er_coef}_cond_length_{config.cond_length}",timestamp)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(checkpoint, os.path.join(save_path, "uper_value_func_checkpoint.pt"))


if __name__ == "__main__":
    train()
