import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)
# edm
from torch import Tensor

"""
根据之前的教训 在写代码之前先列一下框架
"""
def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))

class EDMDiffusion(nn.Module):
    """
    main func: compute_loss sample
    compute_loss{
        1.与之前loss对接
        2.维护内部各项函数与对应参数
    }
    sample{
        1.输入与之前的sample对接
        2.内部函数与参数
    }
    check norm{
        1. 核心就是可以确保loss 和 sample 过程中这个norm的尺度是相同的。
    }
    edm_cfg={
        num_steps_denoising: int
        sigma_min: float = 2e-3
        # simga sampler
        sigma_max: float = 40
        # diffusion sampler
        sigma_max: float = 10
        rho: int = 7
        order: int = 1
        s_churn: float = 0
        s_tmin: float = 0
        s_tmax: float = float("inf")
        s_noise: float = 1
        scale: float = 1.478
        loc: float = -0.225
        sigma_offset_noise: float = 1.0
        sigma_data: float = 1.0
        ·····
    }
    """
    def __init__(self, model, horizon, observation_dim, action_dim, edm_cfg, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1,):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim + 1 # 应为add了reward
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        # edm
        self.inner_model = model
        self.edm_cfg = edm_cfg
        self.device = next(model.parameters()).device
        # sigmas
        self.sigmas = self.build_sigma(
            denoise_steps=self.edm_cfg.num_steps_denoising,
            sigma_min=self.edm_cfg.sigma_min,
            sigma_max=self.edm_cfg.sigma_max_sampler,
            rho=self.edm_cfg.rho,
            device=self.device
        )

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights
    
    #------------------------------------------ edm_compounent ------------------------------------------#
    def sample_sigma_training(self, batch_size, device):
        s = torch.randn(batch_size, device=device) * self.edm_cfg.scale + self.edm_cfg.loc
        return s.exp().clip(self.edm_cfg.sigma_min, self.edm_cfg.sigma_max_training)
    
    def compute_edm_coef(self, sigma):
        sigma = (sigma**2 + self.edm_cfg.sigma_offset_noise**2).sqrt()
        c_in = 1 / (sigma**2 + self.edm_cfg.sigma_data**2).sqrt()
        c_skip = self.edm_cfg.sigma_data**2 / (sigma**2 + self.edm_cfg.sigma_data**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        return *(add_dims(c, 3) for c in (c_in, c_out, c_skip)), add_dims(c_noise, 1)
    
    def denoise_forward(self, noisy_x, sigma, cond_traj, returns):
        c_in, c_out, c_skip, c_noise = self.compute_edm_coef(sigma)
        
        # rescale
        rescale_cond = cond_traj / self.edm_cfg.sigma_data
        rescale_noise = noisy_x * c_in

        # add cond traj (直接cat edm是这样的,但是transformer mask要做好,不然会有点问题)
        cond_length = cond_traj.shape[1]
        rescale_noise = torch.cat([rescale_cond, rescale_noise], dim=1) # shape check

        # model output (returns的格式要注意保持一致)
        out_seq_length = rescale_noise.shape[1] - cond_length
        model_output = self.inner_model(rescale_noise, c_noise, returns)[:,-out_seq_length:]

        # denoised (按理来说 这个是预测的后半段轨迹)
        denoised = model_output * c_out + noisy_x * c_skip
        return model_output, denoised


    def compute_loss(self, x, cond, returns=None):
        """
        x: [batch_size, seq_len, traj_dim]
        cond[0]: [batch_size, cond_length, traj_dim]
        return: [batch_size]
        """
        cond_length = cond[0].shape[1]
        batch_size, seq_len = x.shape[0], x.shape[1]
        device = x.device

        # reshape 因为model预测的是后面那一段，所以进行一个reshape
        x = x[:, cond_length:]
        seq_len = seq_len - cond_length

        #加噪
        sigma = self.sample_sigma_training(batch_size, device)
        _, c_out, c_skip, _ = self.compute_edm_coef(sigma)
        offset_noise = self.edm_cfg.sigma_offset_noise * torch.randn(batch_size, seq_len, 1, device=device)
        noisy_x = x + offset_noise + torch.randn_like(x) * add_dims(sigma, x.ndim) # 等，这个加完噪之后，噪声强度很强的之后肯定要有归一化

        # model output
        model_output, denoised = self.denoise_forward(noisy_x, sigma, cond[0], returns)

        # compute loss
        target = (x - c_skip * noisy_x) / c_out
        loss = F.mse_loss(model_output, target)

        return loss, {"loss_denoising": loss.detach()}
    
    def build_sigma(self, denoise_steps, sigma_min, sigma_max, rho, device):
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        l = torch.linspace(0, 1, denoise_steps, device=device)
        sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat((sigmas, sigmas.new_zeros(1)))


    @torch.no_grad()
    def sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        # 参数准备
        device = cond[0].device
        batch_size, cond_length, traj_dim = cond[0].shape[0], cond[0].shape[1], cond[0].shape[2]
        predict_seq_len = self.horizon - cond_length
        s_in = torch.ones(batch_size, device=device)
        gamma_ = min(self.edm_cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        
        # denoise 可能是只要均值一样
        x = torch.randn(batch_size, predict_seq_len, traj_dim, device=device)

        for sigma, next_sigma in zip(self.sigmas[:-1], self.sigmas[1:]):
            gamma = gamma_ if self.edm_cfg.s_tmin <= sigma <= self.edm_cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(x) * self.edm_cfg.s_noise
                x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5

            _, denoised = self.denoise_forward(x, sigma * s_in, cond[0], returns) # 归一化有没有复原？

            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            if self.edm_cfg.order == 1 or next_sigma == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                _, denoised_2 = self.denoise_forward(x_2, next_sigma * s_in, cond, returns)
                d_2 = (x_2 - denoised_2) / next_sigma
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
        
        # 进行拼接
        total_traj = torch.cat([cond[0], x], dim=1) # 检查归一化是否一致。

        return total_traj



    #------------------------------------------ sampling ------------------------------------------#

#    "" def predict_start_from_noise(self, x_t, t, noise):
#         '''
#             if self.predict_epsilon, model output is (scaled) noise;
#             otherwise, model predicts x0 directly
#         '''
#         if self.predict_epsilon:
#             return (
#                 extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
#                 extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
#             )
#         else:
#             return noise

#     def q_posterior(self, x_start, x_t, t):
#         posterior_mean = (
#             extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
#             extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
#         )
#         posterior_variance = extract(self.posterior_variance, t, x_t.shape)
#         posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
#         return posterior_mean, posterior_variance, posterior_log_variance_clipped

#     def p_mean_variance(self, x, cond, t, returns=None):
#         if self.model.calc_energy:
#             assert self.predict_epsilon
#             x = torch.tensor(x, requires_grad=True)
#             t = torch.tensor(t, dtype=torch.float, requires_grad=True)
#             returns = torch.tensor(returns, requires_grad=True)

#         if self.returns_condition:
#             # epsilon could be epsilon or x0 itself
#             epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
#             epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
#             epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
#         else:
#             epsilon = self.model(x, cond, t)

#         t = t.detach().to(torch.int64)
#         x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

#         if self.clip_denoised:
#             x_recon.clamp_(-1., 1.)
#         else:
#             assert RuntimeError()

#         model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
#                 x_start=x_recon, x_t=x, t=t)
#         return model_mean, posterior_variance, posterior_log_variance

#     @torch.no_grad()
#     def p_sample(self, x, cond, t, returns=None):
#         b, *_, device = *x.shape, x.device
#         model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
#         noise = 0.5*torch.randn_like(x)
#         # no noise when t == 0
#         nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
#         return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

#     @torch.no_grad()
#     def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
#         device = self.betas.device

#         batch_size = shape[0]
#         x = 0.5*torch.randn(shape, device=device)
#         action_reward_dim = self.action_dim + 1
#         x = apply_conditioning(x, cond, action_reward_dim)

#         if return_diffusion: diffusion = [x]

#         progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
#         for i in reversed(range(0, self.n_timesteps)):
#             timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
#             x = self.p_sample(x, cond, timesteps, returns)
#             x = apply_conditioning(x, cond, action_reward_dim)

#             progress.update({'t': i})

#             if return_diffusion: diffusion.append(x)

#         progress.close()

#         if return_diffusion:
#             return x, torch.stack(diffusion, dim=1)
#         else:
#             return x

#     @torch.no_grad()
#     def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
#         '''
#             conditions : [ (time, state), ... ]
#         '''
#         device = self.betas.device
#         batch_size = len(cond[0])
#         horizon = horizon or self.horizon
#         shape = (batch_size, horizon, self.transition_dim)

#         return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

#     def grad_p_sample(self, x, cond, t, returns=None):
#         b, *_, device = *x.shape, x.device
#         model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
#         noise = 0.5*torch.randn_like(x)
#         # no noise when t == 0
#         nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
#         return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

#     def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
#         device = self.betas.device

#         batch_size = shape[0]
#         x = 0.5*torch.randn(shape, device=device)
#         x = apply_conditioning(x, cond, self.action_dim)

#         if return_diffusion: diffusion = [x]

#         progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
#         for i in reversed(range(0, self.n_timesteps)):
#             timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
#             x = self.grad_p_sample(x, cond, timesteps, returns)
#             x = apply_conditioning(x, cond, self.action_dim)

#             progress.update({'t': i})

#             if return_diffusion: diffusion.append(x)

#         progress.close()

#         if return_diffusion:
#             return x, torch.stack(diffusion, dim=1)
#         else:
#             return x

#     def grad_conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
#         '''
#             conditions : [ (time, state), ... ]
#         '''
#         device = self.betas.device
#         batch_size = len(cond[0])
#         horizon = horizon or self.horizon
#         shape = (batch_size, horizon, self.transition_dim)

#         return self.grad_p_sample_loop(shape, cond, returns, *args, **kwargs)

#     #------------------------------------------ training ------------------------------------------#

#     def q_sample(self, x_start, t, noise=None):
#         if noise is None:
#             noise = torch.randn_like(x_start)

#         sample = (
#             extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
#             extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
#         )

#         return sample

#     def p_losses(self, x_start, cond, t, returns=None):
#         cond_length = cond[0].shape[-2]
#         noise = torch.randn_like(x_start)

#         if self.predict_epsilon:
#             # Cause we condition on obs at t=0
#             noise[:,:cond_length,:] = 0
#             # noise[:, 0, self.action_dim:] = 0

#         x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
#         action_reward_dim = self.action_dim + 1
#         x_noisy = apply_conditioning(x_noisy, cond, action_reward_dim)

#         if self.model.calc_energy:
#             assert self.predict_epsilon
#             x_noisy.requires_grad = True
#             t = torch.tensor(t, dtype=torch.float, requires_grad=True)
#             returns.requires_grad = True
#             noise.requires_grad = True

#         x_recon = self.model(x_noisy, cond, t, returns)

#         if not self.predict_epsilon:
#             x_recon = apply_conditioning(x_recon, cond, action_reward_dim)

#         assert noise.shape == x_recon.shape

#         if self.predict_epsilon:
#             loss, info = self.loss_fn(x_recon, noise)
#         else:
#             loss, info = self.loss_fn(x_recon, x_start)
        
#         info["diffusion_loss"] = loss

#         return loss, info""

    def loss(self, x, cond, returns=None):
        # batch_size = len(x)
        # t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        # return self.p_losses(x, cond, t, returns)
        return self.compute_loss(x,cond,returns)

    def forward(self, cond, *args, **kwargs):
        # return self.conditional_sample(cond=cond, *args, **kwargs)
        return self.sample(cond, *args, **kwargs)


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1,):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim + 1 # 应为add了reward
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.model.calc_energy:
            assert self.predict_epsilon
            x = torch.tensor(x, requires_grad=True)
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns = torch.tensor(returns, requires_grad=True)

        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        action_reward_dim = self.action_dim + 1
        x = apply_conditioning(x, cond, action_reward_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, action_reward_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def grad_p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.grad_p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def grad_conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.grad_p_sample_loop(shape, cond, returns, *args, **kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        cond_length = cond[0].shape[-2]
        noise = torch.randn_like(x_start)

        if self.predict_epsilon:
            # Cause we condition on obs at t=0
            noise[:,:cond_length,:] = 0
            # noise[:, 0, self.action_dim:] = 0

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        action_reward_dim = self.action_dim + 1
        x_noisy = apply_conditioning(x_noisy, cond, action_reward_dim)

        if self.model.calc_energy:
            assert self.predict_epsilon
            x_noisy.requires_grad = True
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns.requires_grad = True
            noise.requires_grad = True

        x_recon = self.model(x_noisy, cond, t, returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, action_reward_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)
        
        info["diffusion_loss"] = loss

        return loss, info

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)
    
    
class GaussianInvDynDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True, hidden_dim=256,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1, ar_inv=False, train_only_inv=False):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.ar_inv = ar_inv
        self.train_only_inv = train_only_inv
        if self.ar_inv:
            self.inv_model = ARInvModel(hidden_dim=hidden_dim, observation_dim=observation_dim, action_dim=action_dim)
        else:
            self.inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.action_dim),
            )
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses['state_l2'](loss_weights)

    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = 1
        reward_observation_dim = self.observation_dim + 1 
        dim_weights = torch.ones(reward_observation_dim, dtype=torch.float32)
        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself # 下面就是在计算model free的loss
            
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)
        action_reward_dim = 1
        x = apply_conditioning(x, cond, action_reward_dim)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, action_reward_dim)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        reward_observation_dim = self.observation_dim + 1
        shape = (batch_size, horizon, reward_observation_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        noise = torch.randn_like(x_start) # start 里面只有state序列 cond为每一条轨迹的初始序列

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        action_reward_dim = 1 # 这里action取0，reward取1
        x_noisy = apply_conditioning(x_noisy, cond, action_reward_dim)

        x_recon = self.model(x_noisy, cond, t, returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, action_reward_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond, returns=None):
        action_reward_dim = self.action_dim + 1
        if self.train_only_inv:
            # Calculating inv loss
            x_t = x[:, :-1, action_reward_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, action_reward_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                loss = self.inv_model.calc_loss(x_comb_t, a_t)
                info = {'a0_loss':loss}
            else:
                pred_a_t = self.inv_model(x_comb_t)
                loss = F.mse_loss(pred_a_t, a_t)
                info = {'a0_loss': loss}
        else:
            batch_size = len(x)
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long() # 这个n_timestep 200？要denoise 200次？
            diffuse_loss, info = self.p_losses(x[:, :, self.action_dim:], cond, t, returns) # diffusion loss 把action排除掉了，很合理。就是diffusion在重构噪声。
            # Calculating inv loss
            x_t = x[:, :-1, action_reward_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, action_reward_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1) # 相邻两个action拼接在一起
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                inv_loss = self.inv_model.calc_loss(x_comb_t, a_t)
            else:
                pred_a_t = self.inv_model(x_comb_t)
                inv_loss = F.mse_loss(pred_a_t, a_t)

            loss = (1 / 2) * (diffuse_loss + inv_loss)
            info["diffuse_loss"] = diffuse_loss
            info["inv_loss"] = inv_loss

        return loss, info

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)


class ARInvModel(nn.Module):
    def __init__(self, hidden_dim, observation_dim, action_dim, low_act=-1.0, up_act=1.0):
        super(ARInvModel, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.action_embed_hid = 128
        self.out_lin = 128
        self.num_bins = 80

        self.up_act = up_act
        self.low_act = low_act
        self.bin_size = (self.up_act - self.low_act) / self.num_bins
        self.ce_loss = nn.CrossEntropyLoss()

        self.state_embed = nn.Sequential(
            nn.Linear(2 * self.observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lin_mod = nn.ModuleList([nn.Linear(i, self.out_lin) for i in range(1, self.action_dim)])
        self.act_mod = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, self.action_embed_hid), nn.ReLU(),
                                                    nn.Linear(self.action_embed_hid, self.num_bins))])

        for _ in range(1, self.action_dim):
            self.act_mod.append(
                nn.Sequential(nn.Linear(hidden_dim + self.out_lin, self.action_embed_hid), nn.ReLU(),
                              nn.Linear(self.action_embed_hid, self.num_bins)))

    def forward(self, comb_state, deterministic=False):
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        lp_0 = self.act_mod[0](state_d)
        l_0 = torch.distributions.Categorical(logits=lp_0).sample()

        if deterministic:
            a_0 = self.low_act + (l_0 + 0.5) * self.bin_size
        else:
            a_0 = torch.distributions.Uniform(self.low_act + l_0 * self.bin_size,
                                              self.low_act + (l_0 + 1) * self.bin_size).sample()

        a = [a_0.unsqueeze(1)]

        for i in range(1, self.action_dim):
            lp_i = self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](torch.cat(a, dim=1))], dim=1))
            l_i = torch.distributions.Categorical(logits=lp_i).sample()

            if deterministic:
                a_i = self.low_act + (l_i + 0.5) * self.bin_size
            else:
                a_i = torch.distributions.Uniform(self.low_act + l_i * self.bin_size,
                                                  self.low_act + (l_i + 1) * self.bin_size).sample()

            a.append(a_i.unsqueeze(1))

        return torch.cat(a, dim=1)

    def calc_loss(self, comb_state, action):
        eps = 1e-8
        action = torch.clamp(action, min=self.low_act + eps, max=self.up_act - eps)
        l_action = torch.div((action - self.low_act), self.bin_size, rounding_mode='floor').long()
        state_inp = comb_state

        state_d = self.state_embed(state_inp)
        loss = self.ce_loss(self.act_mod[0](state_d), l_action[:, 0])

        for i in range(1, self.action_dim):
            loss += self.ce_loss(self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](action[:, :i])], dim=1)),
                                     l_action[:, i])

        return loss/self.action_dim


class ActionGaussianDiffusion(nn.Module):
    # Assumes horizon=1
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1,):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.model.calc_energy:
            assert self.predict_epsilon
            x = torch.tensor(x, requires_grad=True)
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns = torch.tensor(returns, requires_grad=True)

        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, returns=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        shape = (batch_size, self.action_dim)
        cond = cond[0]
        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)

    def grad_p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def grad_p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = 0.5*torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    def grad_conditional_sample(self, cond, returns=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        shape = (batch_size, self.action_dim)
        cond = cond[0]
        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, action_start, state, t, returns=None):
        noise = torch.randn_like(action_start)
        action_noisy = self.q_sample(x_start=action_start, t=t, noise=noise)

        if self.model.calc_energy:
            assert self.predict_epsilon
            action_noisy.requires_grad = True
            t = torch.tensor(t, dtype=torch.float, requires_grad=True)
            returns.requires_grad = True
            noise.requires_grad = True

        pred = self.model(action_noisy, state, t, returns)

        assert noise.shape == pred.shape

        if self.predict_epsilon:
            loss = F.mse_loss(pred, noise)
        else:
            loss = F.mse_loss(pred, action_start)

        return loss, {'a0_loss':loss}

    def loss(self, x, cond, returns=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        assert x.shape[1] == 1 # Assumes horizon=1
        x = x[:,0,:]
        cond = x[:,self.action_dim:] # Observation
        x = x[:,:self.action_dim] # Action
        return self.p_losses(x, cond, t, returns)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond=cond, *args, **kwargs)

