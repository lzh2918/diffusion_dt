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
    

class batch_eval_env:
    def __init__(self, eval_env_list):
        self.eval_batch = len(eval_env_list)
        self.eval_env_list = eval_env_list
        self.observation_space = self.eval_env_list[0].observation_space
        self.action_space = self.eval_env_list[0].action_space
        self.last_done = [False for _ in range(self.eval_batch)]
    
    def seed(self, seed):
        for single_eval_env in self.eval_env_list:
            single_eval_env.seed(int(seed))

    def reset(self):
        batch_data = []
        for single_eval_env in self.eval_env_list:
            # 最后的[0]用来降维
            batch_data.append(single_eval_env.reset()[0])
        batch_data = np.array(batch_data)
        self.zero_state = batch_data[0][None]
        self.last_done = [False for _ in range(self.eval_batch)]
        return batch_data
    
    def step(self, actions):
        # 如果单个环境done了，就一直返回全零的内容
        next_state = []
        reward = []
        done = []
        info = {}
        for i in range(self.eval_batch):
            if not self.last_done[i]:
                single_next_state, single_reward, single_done, _ = self.eval_env_list[i].step(actions[i])
                self.last_done[i] = single_done
            else:
                single_next_state, single_reward, single_done = self.zero_state, 0.0, True 
            next_state.append(single_next_state)
            reward.append(single_reward)
            done.append(single_done)
        next_state = np.concatenate(next_state,axis=0)
        reward = np.array(reward)
        done = np.array(done)
        return next_state, reward, done, info




def build_traj(states, actions, norm_dateset, unnorm_coef):
    state_mean = torch.tensor(unnorm_coef[0], dtype=states.dtype,device=states.device).unsqueeze(dim=0)
    state_std = torch.tensor(unnorm_coef[1], dtype=states.dtype,device=states.device).unsqueeze(dim=0)
    states = states * state_std + state_mean

    normed_states = torch.tensor(norm_dateset.normalizer(states.cpu().numpy(), "observations"), dtype=states.dtype,device=states.device)
    normed_actions = torch.tensor(norm_dateset.normalizer(actions.cpu().numpy(), "actions"), dtype=states.dtype,device=states.device)

    traj = torch.cat([normed_actions, normed_states], dim=-1)

    return traj

# Training and evaluation logic
@torch.no_grad()
def eval_rollout(
    model: DecisionTransformer,
    env: batch_eval_env,
    target_return: float,
    if_uper_return: bool,
    uper_vf,
    norm_dataset, # 用来norm
    unnorm_coef, # 用来upf unnorm
    cond_length,
    return_scale, # 和vf对应的return scale
    reward_scale, # DT本身的reward scale
    device: str = "cpu",
) -> Tuple[float, float]:
    eval_batch = env.eval_batch
    states = torch.zeros(
        eval_batch, model.episode_len + 1, model.state_dim, dtype=torch.float, device=device
    )
    actions = torch.zeros(
        eval_batch, model.episode_len, model.action_dim, dtype=torch.float, device=device
    )
    returns = torch.zeros(eval_batch, model.episode_len + 1, dtype=torch.float, device=device)
    time_steps = torch.arange(eval_batch, model.episode_len, dtype=torch.long, device=device).repeat(eval_batch).view(eval_batch,-1)

    states[:, 0] = torch.as_tensor(env.reset(), device=device)
    returns[:, 0] = torch.as_tensor(target_return, device=device)

    target_keep_returns = torch.zeros(eval_batch, model.episode_len + 1, dtype=torch.float, device=device)
    target_keep_returns[:, 0] = torch.as_tensor(target_return, device=device)

    # cannot step higher than model episode len, as timestep embeddings will crash
    episode_return, episode_len = np.zeros((10,)), np.zeros((10,))

    # deal with return to go
    uper_error_list = []

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
        predicted_action = predicted_actions[:, -1].squeeze().cpu().numpy()
        next_state, reward, done, info = env.step(predicted_action)
        # at step t, we predict a_t, get s_{t + 1}, r_{t + 1}
        actions[:, step] = torch.as_tensor(predicted_action, dtype=torch.float, device=device)
        states[:, step + 1] = torch.as_tensor(next_state.astype(np.float), dtype=torch.float, device=device).squeeze()

        reward = torch.as_tensor(reward, dtype=torch.float, device=device)

        # uper_return predict
        # 此处设定就是uper vf预测的是cond length之后下一步的return to go
        if step >= cond_length-1 and if_uper_return:
            traj = build_traj(
                states[:,:step+1][:,-cond_length:],
                actions[:,:step+1][:,-cond_length:],
                norm_dataset,
                unnorm_coef,
            )
            with torch.no_grad():
                uper_returns = uper_vf(
                    traj,
                    time_steps[:,:step+1][:,-cond_length:]
                ).squeeze()
            # 有的环境会提前done掉，这里要维护一下
            uper_returns[done] = 0.0
            uper_returns = uper_returns * return_scale * reward_scale
            uper_error_list.append((uper_returns - (target_keep_returns[:, step] - reward)).cpu().numpy())

            less_index = uper_returns < (target_keep_returns[:, step] - reward)
            uper_returns[less_index] = (target_keep_returns[:, step] - reward)[less_index]
            next_return = uper_returns
        else:
            next_return = returns[:, step] - reward
        next_return[done] = 0
        returns[:, step + 1] = torch.as_tensor(next_return)
        target_keep_returns[:, step + 1] = torch.as_tensor(target_keep_returns[:, step] - reward)

        episode_return += reward.cpu().numpy()
        episode_add = np.ones_like(episode_len)
        episode_add[done] = 0
        episode_len += episode_add

        if done.all():
            break
    if if_uper_return:
        uper_error = np.sum(np.stack(uper_error_list,axis=1), axis=1)/(episode_len-9)
    else:
        uper_error = np.zeros((eval_batch,))

    return episode_return, episode_len, uper_error


# @pyrallis.wrap()
def train(config):
    torch.set_num_threads(4)
    if type(config.target_returns) == str:
        config.target_returns = ast.literal_eval(config.target_returns.replace("_",", "))
    if type(config.dataset_scale) == str:
        config.dataset_scale = ast.literal_eval(config.dataset_scale.replace("_",", "))
    set_seed(config.train_seed, deterministic_torch=config.deterministic_torch)
    # init tensorboard 
    current_upupupup_dir = os.path.dirname(os.path.dirname(current_upup_dir))
    tb_root_dir_path = os.path.join(current_upupupup_dir, "exp_result", "tb", "collect_data")
    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    uper_vf_tag = config.uper_vf_path.split("/")[-3]
    name = "hor_" + str(config.horizon) + "_geper_" + str(config.generate_percentage) + f"_{uper_vf_tag}"
    log_path = os.path.join(tb_root_dir_path, config.project, config.group, config.env_name, name, f"trainseed_{config.train_seed}_evalseed_{config.eval_seed}_"+timestamp)
    print("tensorboard logpath: ", log_path)
    if not os.path.exists(log_path):
            os.makedirs(log_path)
    writer = SummaryWriter(log_dir=log_path)

    # data & dataloader setup
    # dataset = SequenceDataset(
    #     config.env_name, seq_len=config.seq_len, reward_scale=config.reward_scale
    # )
    init_env = gym.make(config.env_name)

    diffusion_databuffer = traj_buffer(
        horizon=config.horizon,
        obs_dim=init_env.observation_space.shape[0],
        action_dim=init_env.action_space.shape[0],
    )

    
    diffusion_databuffer.load(config.diffusion_data_load_path)

    dataset = MixSequenceDataset(
        config.env_name, traj_buffer=diffusion_databuffer, seq_len=config.seq_len, reward_scale=config.reward_scale, gen_per=config.generate_percentage,
    )

    trainloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        num_workers=config.num_workers,
    )

    # evaluation environment with state & reward preprocessing (as in dataset above)
    eval_env = wrap_env(
        env=gym.make(config.env_name),
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    )
    eval_env_list = [wrap_env(
        env=gym.make(config.env_name),
        state_mean=dataset.state_mean,
        state_std=dataset.state_std,
        reward_scale=config.reward_scale,
    ) for i in range(config.eval_batch)]
    batch_env = batch_eval_env(eval_env_list)
    config.state_dim = batch_env.observation_space.shape[0]
    config.action_dim = batch_env.action_space.shape[0]
    # model & optimizer & scheduler setup
    model = DecisionTransformer(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        embedding_dim=config.embedding_dim,
        seq_len=config.seq_len,
        episode_len=config.episode_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        max_action=config.max_action,
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

    # uper value func
    uper_vf = Value_function_Transformer(
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
    uper_vf_state_dict = torch.load(config.uper_vf_path)["model_state"]
    uper_vf.load_state_dict(uper_vf_state_dict)

    norm_dataset = SequenceHalfcondTimestepDataset(
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

    # save config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    trainloader_iter = iter(trainloader)
    for step in trange(config.update_steps, desc="Training"):
        batch = next(trainloader_iter)
        states, actions, returns, time_steps, mask = [b.to(config.device) for b in batch]
        # True value indicates that the corresponding key value will be ignored
        padding_mask = ~mask.to(torch.bool)

        predicted_actions = model(
            states=states,
            actions=actions,
            returns_to_go=returns * config.return_change_coef,
            time_steps=time_steps,
            padding_mask=padding_mask,
        )
        loss = F.mse_loss(predicted_actions, actions.detach(), reduction="none")
        # [batch_size, seq_len, action_dim] * [batch_size, seq_len, 1]
        loss = (loss * mask.unsqueeze(-1)).mean()

        optim.zero_grad()
        loss.backward()
        if config.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        optim.step()
        scheduler.step()

        writer.add_scalar("train/train_loss", loss.item(), step)
        writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], step)

        # validation in the env for the actual online performance
        if step % config.eval_every == 0 or step == config.update_steps - 1:
            model.eval()
            for target_return in config.target_returns:
                for if_uper_return in [True, False]:
                    batch_env.seed(config.eval_seed)
                    eval_returns = []
                    uper_error_list = []
                    for _ in trange(int(config.eval_episodes//batch_env.eval_batch), desc="Evaluation", leave=False):
                        eval_return, eval_len, uper_error = eval_rollout(
                            model=model,
                            env=batch_env,
                            target_return=target_return * config.reward_scale,
                            if_uper_return=if_uper_return,
                            uper_vf=uper_vf,
                            norm_dataset=norm_dataset,
                            unnorm_coef=(dataset.state_mean, dataset.state_std),
                            cond_length=config.cond_length,
                            return_scale=Config.returns_scale,
                            reward_scale=config.reward_scale,
                            device=config.device,
                        )
                        # unscale for logging & correct normalized score computation
                        eval_returns.append(eval_return / config.reward_scale)
                        # uper error
                        uper_error_list.append(uper_error.mean() / config.reward_scale)

                    normalized_scores = (
                        eval_env.get_normalized_score(np.array(eval_returns)) * 100
                    )
                    uper_error_list = np.array(uper_error_list)
                    # writer.add_scalar(f"eval/{target_return}_return_mean", np.mean(eval_returns), step)
                    # writer.add_scalar(f"eval/{target_return}_return_std", np.std(eval_returns), step)
                    writer.add_scalar(f"eval_{target_return}/uper_{if_uper_return}_norm_score_mean", np.mean(normalized_scores), step)
                    writer.add_scalar(f"eval_{target_return}/uper_{if_uper_return}_norm_score_std", np.std(normalized_scores), step)
                    writer.add_scalar(f"eval_{target_return}/uper_{if_uper_return}_uper_error_mean", uper_error_list.mean(), step)
                    writer.add_scalar(f"eval_{target_return}/uper_{if_uper_return}_uper_error_std", uper_error_list.std(), step)
            model.train()

    if config.save_model is not None:
        checkpoint = {
            "model_state": model.state_dict(),
            "state_mean": dataset.state_mean,
            "state_std": dataset.state_std,
        }
        save_root_dir_path = os.path.join(current_upupupup_dir, "exp_result", "saved_model", "collect_data")
        name = "hor_" + str(config.horizon) + "_geper_" + str(config.generate_percentage) + f"_{uper_vf_tag}"
        save_path = os.path.join(save_root_dir_path, config.project, config.group, config.env_name, name, f"trainseed_{config.train_seed}_evalseed_{config.eval_seed}_"+timestamp)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(checkpoint, os.path.join(save_path, "dt_checkpoint.pt"))


if __name__ == "__main__":
    train()
