import torch
import einops

# progress bar
from tqdm.auto import tqdm


class GaussianDiffusion(torch.nn.Module):
    def __init__(
            self,
            model,
            timesteps=2**10,
            ddim_sampling_eta=0
        ):
        super().__init__()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        self.model = model

        # self.loss_function = torch.nn.SmoothL1Loss(reduction='none')
        # self.loss_function = torch.nn.L1Loss(reduction='none')
        self.loss_function = torch.nn.MSELoss(reduction='none')

        # beta schedule
        self.total_timesteps = timesteps
        self.betas = self.sigmoid_beta_schedule(timesteps=self.total_timesteps)

        # alphas
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.nn.functional.pad(input=self.alphas_cumprod[:-1],
                                                           pad=(1, 0),
                                                           value=1)

        # diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

        self.log_one_minus_alphas_cumprod = torch.log(1 - self.alphas_cumprod) # ln(1 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        # posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

        self.posterior_log_variance = torch.log(self.posterior_variance.clamp(min=1e-20)) # ln(posterior_variance)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod)

        self.snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
        self.loss_weight = self.snr / (self.snr + 1)

        self.ddim_sampling_eta = ddim_sampling_eta

        self.to(device=self.device,
                non_blocking=True)


    def linear_beta_schedule(self, timesteps=0, start=0.0001, end=0.02):
        scale = 1000 / timesteps
        beta_start = scale * start
        beta_end = scale * end

        return torch.linspace(start=beta_start,
                              end=beta_end,
                              steps=timesteps,
                              device=self.device)


    def cosine_beta_schedule(self, timesteps=0, start=0.008):
        """
        https://arxiv.org/abs/2102.09672
        """

        steps = timesteps + 1
        interpolation = torch.linspace(start=0,
                                       end=timesteps,
                                       steps=steps,
                                       device=self.device)
        alphas_cumprod = torch.cos((interpolation / timesteps + start) / (1 + start) * torch.pi / 2) ** 2

        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clip(input=betas,
                           min=0,
                           max=0.999)

        return betas


    def sigmoid_beta_schedule(self, timesteps=0, start=-3, end=3, tau=1):
        """
        https://arxiv.org/abs/2212.11972
        """

        steps = timesteps + 1
        interpolation = torch.linspace(start=0,
                                       end=timesteps,
                                       steps=steps,
                                       device=self.device)
        t = interpolation / timesteps
        v_start = torch.tensor(start / tau, device=self.device).sigmoid()
        v_end = torch.tensor(end / tau, device=self.device).sigmoid()

        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)

        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = torch.clip(input=betas,
                           min=0,
                           max=0.999)

        return betas


    def get_index_from_list(self, values, timestep, x_shape):
        """
        returns: the timestep index from a batch of values
        """

        batch_size = timestep.shape[0]
        out = values.gather(dim=-1, index=timestep)
        shape = tuple([batch_size]) + (tuple([1]) * (len(x_shape) - 1)) # tuple of (batch_size, 1, 1, ..., 1)
        out = out.reshape(shape)

        return out


    def q_sample(self, x_initial, timestep_batch):
        """
        x_initial: initial image tensor
        timestep_batch: timesteps
        returns: (noisy image tensor, noise)
        """

        noise = torch.randn_like(x_initial,
                                 device=self.device)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, timestep_batch, x_initial.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, timestep_batch, x_initial.shape)

        return sqrt_alphas_cumprod_t * x_initial + sqrt_one_minus_alphas_cumprod_t * noise, noise


    def predict_start_from_v(self, x_t, t, v):
        return self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t - self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v


    def predict_v(self, x_initial, t, noise):
        return self.get_index_from_list(self.sqrt_alphas_cumprod, t, x_initial.shape) * noise - self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_initial.shape) * x_initial


    def p_losses(self, x_initial, timestep):
        x_noise, noise = self.q_sample(x_initial=x_initial,
                                       timestep_batch=timestep)

        noise_prediction = self.model(x_noise, timestep)

        target = self.predict_v(x_initial=x_initial,
                                t=timestep,
                                noise=noise)

        loss = self.loss_function(input=noise_prediction,
                                  target=target)
        loss = einops.reduce(loss, 'b ... -> b', 'mean')
        loss = loss * self.get_index_from_list(self.loss_weight, timestep, loss.shape)
        return loss.mean()


    def forward(self, x):
        timestep = torch.randint(low=0,
                                 high=self.total_timesteps,
                                 size=tuple([x.shape[0]]),
                                 device=self.device,
                                 dtype=torch.long)

        loss = self.p_losses(x_initial=x, timestep=timestep)

        return loss


    def q_posterior(self, x_initial, x_t, t):
        posterior_mean = self.get_index_from_list(self.posterior_mean_coef1, t, x_t.shape) * x_initial + self.get_index_from_list(self.posterior_mean_coef2, t, x_t.shape) * x_t
        return posterior_mean


    def get_model_prediction(self, image_batch, timestep_batch):
        model_output = self.model(image_batch, timestep_batch)

        x_initial = self.predict_start_from_v(x_t=image_batch,
                                              t=timestep_batch,
                                              v=model_output)

        x_initial = torch.clamp(input=x_initial,
                                min=-1,
                                max=1)

        return x_initial


    def p_sample(self, x, timestep, timestep_index):
        with torch.inference_mode():
            x_initial = self.get_model_prediction(image_batch=x,
                                                  timestep_batch=timestep)

            model_mean = self.q_posterior(x_initial=x_initial,
                                          x_t=x,
                                          t=timestep)

            if timestep_index == 0:
                return model_mean

            noise = torch.randn_like(input=x,
                                     device=self.device)

            model_log_variance = self.get_index_from_list(self.posterior_log_variance, timestep, x.shape)

            return model_mean + (model_log_variance / 2).exp() * noise


    def sample(self, image_size=(64, 64), batch_size=1, return_all_timesteps=False):
        with torch.inference_mode():
            tensor_shape = (batch_size, self.model.image_channels, image_size[0], image_size[1]) # batch size, color channels, height, width

            # sample noise
            image_batch = torch.randn(size=tensor_shape,
                                      device=self.device)

            images = [image_batch]

            for index in tqdm(reversed(range(0, self.total_timesteps)), total=self.total_timesteps):
                timestep_batch = torch.full(size=tuple([batch_size]),
                                        fill_value=index,
                                        device=self.device,
                                        dtype=torch.long)

                image_batch = self.p_sample(image_batch, timestep_batch, index)

                if return_all_timesteps:
                    images.append(image_batch)

            if not return_all_timesteps:
                images.append(image_batch)

            return images


    def predict_noise_from_start(self, x_t, t, x_initial):
        return (self.get_index_from_list(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_initial) / self.get_index_from_list(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)


    def ddim_sample(self, sampling_timesteps=2**8, image_size=(64, 64), batch_size=1, return_all_timesteps=False):
        """
        https://arxiv.org/abs/2010.02502
        """

        with torch.inference_mode():
            times = torch.linspace(start=-1,
                                end=self.total_timesteps - 1,
                                steps=sampling_timesteps + 1,
                                device=self.device) # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

            tensor_shape = (batch_size, self.model.image_channels, image_size[0], image_size[1]) # batch size, color channels, height, width
            image_batch = torch.randn(size=tensor_shape,
                                    device=self.device)
            images = [image_batch]

            for time, next_time in tqdm(time_pairs):
                timestep_batch = torch.full(size=tuple([batch_size]),
                                    fill_value=time,
                                    device=self.device,
                                    dtype=torch.long)

                x_initial = self.get_model_prediction(image_batch=image_batch,
                                                      timestep_batch=timestep_batch)
                predicted_noise = self.predict_noise_from_start(x_t=image_batch,
                                                        t=timestep_batch,
                                                        x_initial=x_initial)

                if next_time < 0:
                    image_batch = x_initial
                else:
                    alpha = self.alphas_cumprod[time]
                    alpha_next = self.alphas_cumprod[next_time]

                    sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                    c = (1 - alpha_next - sigma ** 2).sqrt()

                    noise = torch.randn_like(input=image_batch,
                                            device=self.device)

                    image_batch = x_initial * alpha_next.sqrt() + c * predicted_noise + sigma * noise

                if return_all_timesteps:
                    images.append(image_batch)

            if not return_all_timesteps:
                images.append(image_batch)

            return images
