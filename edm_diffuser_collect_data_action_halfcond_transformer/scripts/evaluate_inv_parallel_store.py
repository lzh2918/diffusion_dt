import diffuser.utils as utils
from ml_logger import logger
import torch
from copy import deepcopy
import numpy as np
import os
import gym
from config.locomotion_config import Config
from diffuser.utils.arrays import to_torch, to_np, to_device
from diffuser.datasets.d4rl import suppress_output
from my_add_code.traj_buffer import traj_buffer
import datetime


def acumulate_reward(reward):
    acumulated_reward = np.zeros_like(reward)
    for i in range(reward.shape[1]):
        if i > 0:
            acumulated_reward[:,i] = reward[:,i-1] + acumulated_reward[:,i-1]
    return acumulated_reward

def cycle(dl):
    while True:
        for data in dl:
            yield data


def get_conditions(batch, action_dim, device):
    cond_length = int(batch.trajectories.shape[-2]/2)
    half_traj_cond = to_torch(batch.trajectories[:, :cond_length, :], device=device) # batch_size, traj_len
    conditions = {0:half_traj_cond}
    returns = batch.returns
    return conditions, returns

def evaluate(args, **deps):
    from ml_logger import logger, RUN
    from config.locomotion_config import Config

    RUN._update(deps)
    Config._update(deps)

    logger.remove('*.pkl')
    logger.remove("traceback.err")
    logger.log_params(Config=vars(Config), RUN=vars(RUN))

    Config.device = 'cuda'
    Config.horizon = args.horizon
    Config.dataset = args.task
    Config.discount = args.discount
    Config.diffusion = args.diffusion
    Config.loader = args.data_loader
    # Config.ar_inv = args.ar_inv
    loadpath = args.loadpath
    return_scale_high = args.return_scale_high
    return_scale_low = args.return_scale_low

    if Config.predict_epsilon:
        prefix = f'predict_epsilon_{Config.n_diffusion_steps}_1000000.0'
    else:
        prefix = f'predict_x0_{Config.n_diffusion_steps}_1000000.0'

    # loadpath = os.path.join(Config.bucket, logger.prefix, 'checkpoint')
    # loadpath = "/data/user/liuzhihong/paper/big_model/diffusion/exp_result/decision_diffuser_collect_data/first_edition/hopper-medium-v2/horizon_20/24-0516-213306/checkpoint"
    
    if Config.save_checkpoints:
        loadpath = os.path.join(loadpath, f'state_{self.step}.pt')
    else:
        loadpath = os.path.join(loadpath, 'state.pt')
    
    state_dict = torch.load(loadpath, map_location=Config.device)

    # Load configs
    torch.backends.cudnn.benchmark = True
    utils.set_seed(Config.seed)

    dataset_config = utils.Config(
        args.data_loader,
        savepath='dataset_config.pkl',
        env=Config.dataset,
        horizon=Config.horizon,
        normalizer=Config.normalizer,
        preprocess_fns=Config.preprocess_fns,
        use_padding=Config.use_padding,
        max_path_length=Config.max_path_length,
        include_returns=Config.include_returns,
        returns_scale=Config.returns_scale,
        discount=Config.discount,
    )

    render_config = utils.Config(
        Config.renderer,
        savepath='render_config.pkl',
        env=Config.dataset,
    )

    dataset = dataset_config()
    renderer = render_config()

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    if Config.diffusion == 'models.GaussianInvDynDiffusion':
        transition_dim = observation_dim
    else:
        transition_dim = observation_dim + action_dim

    model_config = utils.Config(
        Config.model,
        savepath='model_config.pkl',
        horizon=Config.horizon,
        transition_dim=transition_dim + 1, # 因为加了一维reward上去
        cond_dim=observation_dim,
        dim_mults=Config.dim_mults,
        dim=Config.dim,
        returns_condition=Config.returns_condition,
        device=Config.device,
    )

    diffusion_config = utils.Config(
            Config.diffusion,
            savepath='diffusion_config.pkl',
            horizon=Config.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=Config.n_diffusion_steps,
            loss_type=Config.loss_type,
            clip_denoised=Config.clip_denoised,
            predict_epsilon=Config.predict_epsilon,
            ## loss weighting
            action_weight=Config.action_weight,
            loss_weights=Config.loss_weights,
            loss_discount=Config.loss_discount,
            returns_condition=Config.returns_condition,
            condition_guidance_w=Config.condition_guidance_w,
            device=Config.device,
        )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath='trainer_config.pkl',
        train_batch_size=Config.batch_size,
        train_lr=Config.learning_rate,
        gradient_accumulate_every=Config.gradient_accumulate_every,
        ema_decay=Config.ema_decay,
        sample_freq=Config.sample_freq,
        save_freq=Config.save_freq,
        log_freq=Config.log_freq,
        label_freq=int(Config.n_train_steps // Config.n_saves),
        save_parallel=Config.save_parallel,
        bucket=Config.bucket,
        n_reference=Config.n_reference,
        train_device=Config.device,
    )

    model = model_config()
    diffusion = diffusion_config(model)
    trainer = trainer_config(diffusion, dataset, renderer)
    logger.print(utils.report_parameters(model), color='green')
    trainer.step = state_dict['step']
    trainer.model.load_state_dict(state_dict['model'])
    trainer.ema_model.load_state_dict(state_dict['ema'])

    num_eval = 200
    device = Config.device

    # env_list = [gym.make(Config.dataset) for _ in range(num_eval)]
    dones = [0 for _ in range(num_eval)]
    episode_rewards = [0 for _ in range(num_eval)]

    assert trainer.ema_model.condition_guidance_w == Config.condition_guidance_w
    returns = to_device(Config.test_ret * torch.ones(num_eval, 1), device)

    t = 0
    # obs_list = [env.reset()[None] for env in env_list]
    # obs = np.concatenate(obs_list, axis=0)
    # recorded_obs = [deepcopy(obs[:, None])]

    # traj_buffer
    store_buffer = traj_buffer(
        horizon=Config.horizon,
        obs_dim=observation_dim,
        action_dim=action_dim
    )

    # 获得条件的方式修改，和sample轨迹的方式一样，然后去最开始的state作为条件
    condition_data_loader = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=num_eval, num_workers=0, shuffle=True, pin_memory=True
        ))
    
    collected_num = 0
    total_num = args.collect_num

    while collected_num < total_num:
        batch = next(condition_data_loader)
        conditions, dataset_returns = get_conditions(batch, action_dim, device)
        timesteps = batch.timesteps
        random_scale = np.random.random(dataset_returns.shape).astype("float32")
        return_scale = return_scale_low + (return_scale_high - return_scale_low) * random_scale
        dataset_returns = dataset_returns * return_scale
        samples = trainer.ema_model.conditional_sample(conditions, returns=dataset_returns.to(device))
        action_reward_dim = action_dim + 1
        obs_comb = torch.cat([samples[:, :-1, action_reward_dim:], samples[:, 1:, action_reward_dim:]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2*observation_dim)
        # get action
        # action = trainer.ema_model.inv_model(obs_comb).reshape(samples.shape[0], samples.shape[1]-1,-1)
        action = samples[:,:,:action_dim]
        action = to_np(action)
        action = dataset.normalizer.unnormalize(action, 'actions')
        # action = np.pad(action, ((0,0),(0,1),(0,0)), "constant")
        # get reward
        reward = samples[:,:,action_dim]
        reward = to_np(reward)
        # unnorm samples
        obs = to_np(samples[:,:,action_reward_dim:])
        obs = dataset.normalizer.unnormalize(obs, 'observations')
        # unnorm returns
        unnormed_returns = to_np(dataset_returns * Config.returns_scale)
        acumulated_reward = acumulate_reward(reward)
        unnormed_returns = unnormed_returns - acumulated_reward

        # store
        store_buffer.add_batch_data(obs=obs, action=action, reward=reward, returns=unnormed_returns,time_steps=timesteps)

        collected_num += num_eval

    # dir_path = "/data/user/liuzhihong/paper/big_model/diffusion/decision_diffuser_transformer_2/my_add_code/save_traj

    store_buffer.save(args.save_traj_path)
