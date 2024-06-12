import numpy as np
import os
from typing import Tuple, Dict
import torch


class traj_buffer:
    def __init__(
        self,
        horizon,
        obs_dim,
        action_dim,
        max_length = 100000,
    ) -> None:
        self.max_length = max_length
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.size = 0
        self.pos = 0
        self.obs_buffer = np.zeros((max_length, horizon, obs_dim))
        self.action_buffer = np.zeros((max_length, horizon, action_dim))
        self.reward_buffer = np.zeros((max_length, horizon))
        self.return_buffer = np.zeros((max_length, horizon))
        self.timestep_buffer = np.zeros((max_length,))
        self.penalty_buffer = np.zeros((max_length, horizon))

    def add_obs(self, data):
        batch_size = len(data)
        assert data.shape[1] == self.horizon
        if self.pos + batch_size < self.max_length:
            self.obs_buffer[self.pos:self.pos+batch_size] = data
            self.size += batch_size
            self.pos += batch_size
        else: 
            print("buffer is full")
            raise BufferError("buffer is full")
    
    def add_batch_data(self, obs, action, reward, returns, time_steps):
        batch_size = len(obs)
        assert obs.shape[1] == self.horizon
        assert action.shape[1] == self.horizon
        assert reward.shape[1] == self.horizon
        if self.pos + batch_size < self.max_length:
            self.obs_buffer[self.pos:self.pos + batch_size] = obs
            self.action_buffer[self.pos:self.pos+batch_size] = action
            self.reward_buffer[self.pos:self.pos+batch_size] = reward
            self.return_buffer[self.pos:self.pos+batch_size] = returns
            self.timestep_buffer[self.pos:self.pos+batch_size] = time_steps
            self.size += batch_size
            self.pos += batch_size
        else:
            print("buffer is full")

    def random_sample(self, batch_size):
        index = np.random.randint(0,self.size, batch_size)
        traj = {
            "obs": self.obs_buffer[index],
            "action": self.action_buffer[index],
            "reward": self.reward_buffer[index]
        }
        return traj, index
    
    def sample_one(self, reward_scale, obs_mean, obs_std):
        index = np.random.randint(0,self.size, 1)[0]
        obs = self.obs_buffer[index]
        action = self.action_buffer[index]
        returns = self.return_buffer[index] * reward_scale
        time_steps = self.timestep_buffer[index]
        time_steps = np.arange(time_steps, time_steps+self.horizon)
        mask = np.ones(obs.shape[0]) 
        # mask[-1] = 0
        obs = (obs - obs_mean)/obs_std
        return obs.astype(np.float32), action.astype(np.float32), returns.astype(np.float32), time_steps.astype(int), mask.astype(np.float32)
    
    def sample_batch(self, batch_size,reward_scale=1):
        index = np.random.randint(0,self.size, batch_size)
        obs = self.obs_buffer[index]
        action = self.action_buffer[index]
        returns = self.return_buffer[index] * reward_scale
        time_steps = self.timestep_buffer[index]
        # time_steps = np.arange(time_steps, time_steps+self.horizon)
        mask = np.ones(obs.shape[0]) 
        mask[-1] = 0
        return obs.astype(np.float32), action.astype(np.float32), returns.astype(np.float32), time_steps.astype(int), mask.astype(np.float32)

    def sample_all(self):
        traj = {
            "obs": self.obs_buffer[:self.size],
            "action": self.action_buffer[:self.size],
            "reward": self.reward_buffer[:self.size],
            "returns": self.return_buffer[:self.size],
            "timesteps": self.timestep_buffer[:self.size],
        }
        return traj

    def add_action(self, action, index):
        self.action_buffer[index] = action

    def add_reward(self, reward, penalty, index):
        reward = reward.squeeze()
        penalty = penalty.squeeze()
        self.reward_buffer[index] = reward
        self.penalty_buffer[index] = penalty

    def save(self, save_path, processed=False):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        traj = self.sample_all()
        if processed:
            save_path = os.path.join(save_path, "save_traj_total")
        else:
            save_path = os.path.join(save_path, "save_traj")
        np.save(save_path, traj)

    def load(self, load_path):
        data = np.load(load_path, allow_pickle=True)
        data = data.tolist()
        data_size = len(data["obs"])
        self.size = data_size
        self.pos = data_size
        self.obs_buffer[:data_size] = data["obs"]
        self.action_buffer[:data_size] = data["action"]
        self.reward_buffer[:data_size] = data["reward"]
        self.return_buffer[:data_size] = data["returns"]
        self.timestep_buffer[:data_size] = data["timesteps"]


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations)
        self._size = len(observations)
     
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
        }
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }
    

def qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)
    
    has_next_obs = True if 'next_observations' in dataset.keys() else False

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        if has_next_obs:
            new_obs = dataset['next_observations'][i].astype(np.float32)
        else:
            new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue  
        if done_bool or final_timestep:
            episode_step = 0
            if not has_next_obs:
                continue

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }
