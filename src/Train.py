# training
from accelerate import Accelerator, DataLoaderConfiguration
from pathlib import Path
import itertools
import torch
import math

# progress bar
from timeit import default_timer
from tqdm.auto import tqdm

# data visualization
import torchvision
import matplotlib.pyplot


class Train():
    def __init__(
            self,
            diffusion_model,
            data_loader,
            gradient_accumulate_every=1,
            learning_rate=0.001,
            training_epochs=100000,
            adam_betas=(0.9, 0.99),
            sample_every=10000,
            save_every=10,
            sample_batch_size=4,
            sample_batch_image_size=(64, 64),
            results_directory='./results',
            amp=False, # enable mixed precision
            mixed_precision_type='fp16',
            split_batches=True,
            max_grad_norm=1,
            save_latest_only=False,
            minimum_dataset_size=100,
            effective_batch_size=16,
            use_ddim_sampling=True,
            sampling_timesteps=2**8
        ):
        super().__init__()

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        # accelerator
        if not amp:
            mixed_precision_type = 'no'

        dataloader_config = DataLoaderConfiguration(split_batches=split_batches,
                                                    non_blocking=True)
        self.accelerator = Accelerator(dataloader_config=dataloader_config,
                                       mixed_precision=mixed_precision_type)

        self.sample_batch_size = sample_batch_size
        self.sample_batch_image_size = sample_batch_image_size
        self.sample_every = sample_every
        self.save_every = save_every
        self.save_latest_only = save_latest_only
        self.use_ddim_sampling = use_ddim_sampling

        self.data_loader = data_loader
        if len(self.data_loader.dataset) < minimum_dataset_size:
            raise Exception(f'dataset size of {len(self.data_loader.dataset)} is < {minimum_dataset_size}')

        self.gradient_accumulate_every = gradient_accumulate_every
        if not self.gradient_accumulate_every * self.data_loader.batch_size >= effective_batch_size:
            self.gradient_accumulate_every = math.ceil(effective_batch_size / self.data_loader.batch_size)

        self.max_grad_norm = max_grad_norm

        self.training_epochs = training_epochs

        self.model = diffusion_model

        self.sampling_timesteps = sampling_timesteps
        if self.sampling_timesteps > self.model.total_timesteps:
            self.sampling_timesteps = self.model.total_timesteps // 2

        # optimizer
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate,
                                          betas=adam_betas)

        self.results_directory = Path(results_directory)
        self.results_directory.mkdir(exist_ok=True)

        # epoch counter
        self.current_epoch = 0

        # prepare model, dataloader, and optimizer with accelerator
        self.model, self.data_loader, self.optimizer = self.accelerator.prepare(self.model, self.data_loader, self.optimizer)

        self.data_loader = itertools.cycle(self.data_loader)

        self.training_loss_values = []
        self.total_training_time = 0 # seconds


    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.current_epoch,
            'model': self.accelerator.get_state_dict(self.model),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if self.accelerator.scaler is not None else None,
            'training_loss_values': self.training_loss_values,
            'total_training_time': self.total_training_time
        }

        torch.save(data, str(self.results_directory / f'diffusion-{milestone}.pt'))


    def save_point(self):
        if self.save_latest_only:
            self.save('latest')
        else:
            self.save(self.current_epoch)


    def load(self, milestone):
        data = torch.load(str(self.results_directory / f'diffusion-{milestone}.pt'), map_location=self.device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.optimizer.load_state_dict(data['optimizer'])

        if self.accelerator.scaler is not None and data['scaler'] is not None:
            self.accelerator.scaler.load_state_dict(data['scaler'])

        self.current_epoch = data['step']
        self.training_loss_values = data['training_loss_values']
        self.total_training_time = data['total_training_time']

        model.eval()

        model.to(device=self.device,
                 non_blocking=True)


    def train(self):
        self.model.train()

        with tqdm(initial=self.current_epoch,
                  total=self.training_epochs,
                  disable=not self.accelerator.is_main_process) as progress_bar:

            while self.current_epoch < self.training_epochs:
                self.optimizer.zero_grad()

                total_loss = 0

                start_time = default_timer()

                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.data_loader)
                    batch = batch.to(device=self.device,
                                     non_blocking=True)

                    with self.accelerator.autocast():
                        loss = self.model(batch)

                        if math.isnan(loss):
                            self.save_point()
                            print('NAN detected. Turn off AMP to continue training.')
                            return

                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                self.accelerator.clip_grad_norm_(parameters=self.model.parameters(),
                                                 max_norm=self.max_grad_norm)

                self.accelerator.wait_for_everyone()

                self.optimizer.step()

                self.accelerator.wait_for_everyone()

                end_time = default_timer()
                self.total_training_time += end_time - start_time

                progress_bar.set_description(f'loss: {total_loss:.5f}')

                self.training_loss_values.append(total_loss)

                self.current_epoch += 1

                if self.accelerator.is_main_process:
                    if self.current_epoch % self.save_every == 0:
                        self.save_point()

                        self.plot_loss(name=str(self.results_directory / 'loss.png'))

                    if self.current_epoch % self.sample_every == 0:
                        self.sample()

                progress_bar.update(1)


    def plot_loss(self, name = 'loss.png'):
        training_loss = torch.tensor(self.training_loss_values).numpy()
        figure = matplotlib.pyplot.figure(figsize=(10, 7))
        matplotlib.pyplot.title('loss curve')
        matplotlib.pyplot.xlabel('iterations')
        matplotlib.pyplot.ylabel('loss')
        training_iterations = range(len(training_loss))
        matplotlib.pyplot.plot(training_iterations, training_loss, label='training loss')
        figure.legend()
        figure.savefig(name)

        matplotlib.pyplot.close()


    def sample(self, name=None):
        self.model.eval()

        all_images = None

        with torch.inference_mode():
            if self.use_ddim_sampling:
                all_images = self.model.ddim_sample(batch_size=self.sample_batch_size,
                                                    image_size=self.sample_batch_image_size,
                                                    sampling_timesteps=self.sampling_timesteps)
            else:
                all_images = self.model.sample(batch_size=self.sample_batch_size,
                                                    image_size=self.sample_batch_image_size)

        self.model.train()

        all_images = all_images[-1] # get the last images
        all_images = (all_images + 1) / 2 # unnormalize

        if name is None:
            name = self.current_epoch
        else:
            name = f'{self.current_epoch}-{name}'
        torchvision.utils.save_image(all_images, str(self.results_directory / f'sample-{name}.png'))
