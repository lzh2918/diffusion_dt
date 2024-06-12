import sys
sys.path.append("/data/user/liuzhihong/paper/offline/OfflineRL-Kit-main")

import torch
import numpy
from torch import nn
from typing import Dict, List, Union, Optional
import numpy as np
import torch.functional as F
from torch.optim import Adam
import os
from offlinerlkit.utils.logger import Logger


def soft_clamp(
    x : torch.Tensor,
    _min: Optional[torch.Tensor] = None,
    _max: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class EnsembleLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        weight_decay: float = 0.0
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble

        self.register_parameter("weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))

        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))

        self.register_parameter("saved_weight", nn.Parameter(self.weight.detach().clone()))
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias

        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias

        return x

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = self.weight_decay * (0.5*((self.weight**2).sum()))
        return decay_loss


class EnsembleRewardModel(nn.Module):
    def __init__(
        self,
        observation_dim, 
        penalty_coef, 
        num_ensemble: int = 7,
        num_elites: int = 5,
        hidden_dims = [256,256,256],
        weight_decays = None,
        activation = nn.Relu(),
        device: str = "cpu"
    ):
        super(EnsembleRewardModel, self).__init__()
        self.observation_dim = observation_dim
        self.penalty_coef = penalty_coef
        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self.device = torch.device(device)
        self.activation = activation

        # 这个ensemble model怎么建立的，之前不太会，现在可以学习一下了。
        module_list = []
        hidden_dims = [2*self.observation_dim] + list(hidden_dims)
        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
        for in_dim, out_dim, weigth_decay in zip(hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1]):
            module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weigth_decay))
        self.backbones = nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(
            hidden_dims[-1],
            2 * 1, # 一半是奖励，另一半是logvar
            num_ensemble,
            weight_decays[-1]
        )

        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(2) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(2) * -10, requires_grad=True)
        )

        self.register_parameter(
            "elites",
            nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False)
        )

        self.to(self.device)
        

    def forward(self, comb_state):
        comb_state = torch.as_tensor(comb_state, dtype=torch.float32).to(self.device)
        output = self.comb_state
        for layer in self.backbones:
            output = self.activation(layer(output))
        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)

        return mean, logvar
    
    #*********************************原始model的************************
    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.backbones:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
    
    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs
    #*******************************************************************
    
    def get_reward(
        self,
        comb_state: np.ndarray,
    ):
        batch_size = len(comb_state)
        mean, logvar = self.forward(comb_state)

        if self.penalty_coef:
            penalty = np.sqrt(mean.var(0).mean(1)) # 这个维度不太确定，到时候调试一下
        
        # 还是从elite里面取模型比较好，感觉是有点探索性。取平均得到话可能太单调了
        model_idxs = self.random_elite_idxs(batch_size)
        samples = mean[model_idxs, np.arange(batch_size)]

        reward = samples
        reward_penaltied = samples - self.penalty_coef * penalty

        return reward, reward_penaltied


class reward_trainer:
    def __init__(
        self,
        ensemble_model,
        optim,
        scaler,
    ) -> None:
        self.ensemble_model = ensemble_model
        self.optim = optim
        self.scaler = scaler
        self.device = self.ensemble_model.device

    def learn(
        self, 
        comb_state, 
        real_reward,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01,
    ):
        self.ensemble_model.train()
        train_size = comb_state.shape[1] # 这个什么意思？这个第一维是什么东西，不太清楚
        losses = []

        for batch_num in range(int(np.ceil(train_size/batch_size))):
            comb_state_batch = comb_state[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            target_batch = real_reward[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            target_batch = torch.as_tensor(target_batch).to(self.ensemble_model.device)

            reward_pred, logvar  = self.ensemble_model.forward(comb_state_batch)
            inv_var = torch.exp(-logvar)
            # 这个dim可能有点问题，倒是后调试的时候在调整吧。
            mse_loss_inv = (torch.pow(reward_pred - real_reward, 2) * inv_var).mean() 
            var_loss = logvar.mean()
            loss = mse_loss_inv + var_loss
            loss = loss + self.ensemble_model.get_decay_loss()
            loss = loss + logvar_loss_coef * self.ensemble_model.max_logvar.sum() - logvar_loss_coef * self.ensemble_model.min_logvar.sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())

        return np.mean(losses)
    
    def format_samples_for_training(self, data: Dict):
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        comb_state = np.concatenate((obss, next_obss), axis=-1)
        return comb_state, rewards

    # 如果你连跟谁对齐都没决定好，就去写这个代码，怎么可能顺利呢？
    def train(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01
    ) -> None:
        comb_state, rewards = self.format_samples_for_training(data)
        data_size = comb_state.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = comb_state[train_splits.indices], rewards[train_splits.indices]
        holdout_inputs, holdout_targets = comb_state[holdout_splits.indices], rewards[holdout_splits.indices]

        self.scaler.fit(train_inputs) # scaler 要的改
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        holdout_losses = [1e10 for i in range(self.ensemble_model.num_ensemble)]
        # 相当于直接通过ensemble复制了。
        data_idxes = np.random.randint(train_size, size=[self.ensemble_model.num_ensemble, train_size])
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        print("model device: ", self.ensemble_model.device)
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], batch_size, logvar_loss_coef)
            print("model device: ", self.ensemble_model.device)
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.ensemble_model.num_elites]).mean()
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.ensemble_model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        indexes = self.select_elites(holdout_losses)
        self.ensemble_model.set_elites(indexes)
        self.ensemble_model.load_save()
        self.save(logger.model_dir)
        self.ensemble_model.eval()
        logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.ensemble_model.num_elites]).mean()))
    
    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.ensemble_model.eval()
        targets = torch.as_tensor(targets).to(self.ensemble_model.device)
        mean, _ = self.ensemble_model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2)) # 感觉是这个dim有点问题，先调试一下
        val_loss = list(loss.cpu().numpy())
        return val_loss
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)