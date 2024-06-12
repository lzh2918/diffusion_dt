import argparse
import os
import sys
sys.path.append("/data/user/liuzhihong/paper/offline/OfflineRL-Kit-main")
import random

import gym
import d4rl
import d4rl.gym_mujoco

import numpy as np
import torch


from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, EnsembleDynamicsModel
# from offlinerlkit.dynamics import EnsembleDynamics
# from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.termination_fns import get_termination_fn
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger
# from offlinerlkit.policy_trainer import MBPolicyTrainer
# from offlinerlkit.policy import COMBOPolicy

from ensemble_dynamics_classify import EnsembleDynamics
from mb_policy_trainer_cycle import MBPolicyTrainer
from combo_classify import COMBOPolicy
from buffer_classify import ReplayBuffer_classify
from logger import make_log_dirs

from utils.reward_model import EnsembleRewardModel, reward_trainer
from utils.scaler_change import StandardScaler


"""
suggested hypers

halfcheetah-medium-v2: rollout-length=5, cql-weight=0.5
hopper-medium-v2: rollout-length=5, cql-weight=5.0
walker2d-medium-v2: rollout-length=1, cql-weight=5.0
halfcheetah-medium-replay-v2: rollout-length=5, cql-weight=0.5
hopper-medium-replay-v2: rollout-length=5, cql-weight=0.5
walker2d-medium-replay-v2: rollout-length=1, cql-weight=0.5
halfcheetah-medium-expert-v2: rollout-length=5, cql-weight=5.0
hopper-medium-expert-v2: rollout-length=5, cql-weight=5.0
walker2d-medium-expert-v2: rollout-length=1, cql-weight=5.0
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="/data/user/liuzhihong/paper/offline/exp_result")
    parser.add_argument("--exp_group", type=str, default="test")
    parser.add_argument("--algo-name", type=str, default="combo")
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=int(np.random.randint(0,50,1)[0]))
    parser.add_argument("--cycle_threshold", type=float, default=2.0) # new add
    parser.add_argument("--grad_max_norm", type=float, default=1.0) # new add
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-alpha", default=True)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    parser.add_argument("--uniform-rollout", type=bool, default=False)
    parser.add_argument("--rho-s", type=str, default="mix", choices=["model", "mix"])

    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--load-dynamics-path", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def train(args=get_args()):
    # create env and dataset\
    torch.set_num_threads(8)
    env = gym.make(args.task)
    dataset = qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, dist, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.epoch)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create dynamics
    load_dynamics_model = True if args.load_dynamics_path else False
    dynamics_model = EnsembleRewardModel(
        obs_dim=np.prod(args.obs_shape),
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=args.device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )

    scaler = StandardScaler()
    # 仅为调试，所以写成这样。
    # scaler.load_scaler("/data/user/liuzhihong/paper/offline/OfflineRL-Kit-main/exp_result/hopper-medium-v2/mopo&penalty_coef=5.0&rollout_length=5/seed_1&timestamp_24-0129-210756/model")

    termination_fn = get_termination_fn(task=args.task)
    dynamics = reward_trainer(
        dynamics_model,
        dynamics_optim,
        scaler,
    )

    if args.load_dynamics_path:
        dynamics.load(args.load_dynamics_path)

    # create policy
    # policy = COMBOPolicy(
    #     dynamics,
    #     actor,
    #     critic1,
    #     critic2,
    #     actor_optim,
    #     critic1_optim,
    #     critic2_optim,
    #     action_space=env.action_space,
    #     cycle_threshold=args.cycle_threshold, # new add
    #     grad_max_norm=args.grad_max_norm, # new add
    #     tau=args.tau,
    #     gamma=args.gamma,
    #     alpha=alpha,
    #     cql_weight=args.cql_weight,
    #     temperature=args.temperature,
    #     max_q_backup=args.max_q_backup,
    #     deterministic_backup=args.deterministic_backup,
    #     with_lagrange=args.with_lagrange,
    #     lagrange_threshold=args.lagrange_threshold,
    #     cql_alpha_lr=args.cql_alpha_lr,
    #     num_repeart_actions=args.num_repeat_actions,
    #     uniform_rollout=args.uniform_rollout,
    #     rho_s=args.rho_s
    # )

    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    real_buffer.load_dataset(dataset)
    # fake_buffer = ReplayBuffer_classify(
    #     buffer_size=args.rollout_batch_size*args.rollout_length*args.model_retain_epochs,
    #     obs_shape=args.obs_shape,
    #     obs_dtype=np.float32,
    #     action_dim=args.action_dim,
    #     action_dtype=np.float32,
    #     device=args.device
    # )

    # log
    logdir = os.path.join(args.logdir, args.exp_group)
    log_dirs = make_log_dirs(logdir, args.task, args.algo_name, args.seed, vars(args), record_params=["cql_weight", "rollout_length", "cycle_threshold", "grad_max_norm"])
    # key: output file name, value: output handler type"
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # create policy trainer
    # policy_trainer = MBPolicyTrainer(
    #     policy=policy,
    #     eval_env=env,
    #     real_buffer=real_buffer,
    #     fake_buffer=fake_buffer,
    #     logger=logger,
    #     rollout_setting=(args.rollout_freq, args.rollout_batch_size, args.rollout_length),
    #     epoch=args.epoch,
    #     step_per_epoch=args.step_per_epoch,
    #     batch_size=args.batch_size,
    #     real_ratio=args.real_ratio,
    #     eval_episodes=args.eval_episodes,
    #     lr_scheduler=lr_scheduler
    # )

    # train
    if not load_dynamics_model:
        dynamics.train(real_buffer.sample_all(), logger, max_epochs_since_update=5)
    
    # policy_trainer.train()


if __name__ == "__main__":
    train()