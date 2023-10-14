# use non-interactive backend to avoid
# RuntimeError: main thread is not in main loop
# with tkinter
import matplotlib
matplotlib.use('agg') # raster graphics, png

# neural network
import torch
from UNet import UNet
from GaussianDiffusion import GaussianDiffusion

# training
from Train import Train

# architecture visualization
from torch.utils.tensorboard import SummaryWriter

# progress bar
from tqdm.auto import tqdm

# image dataset
import torchvision
import matplotlib.pyplot
from torch.utils.data import DataLoader
import numpy
from PIL import Image
from ImageDataset import ImageDataset
import os

# gif
import matplotlib.animation


device = 'cpu'

if torch.cuda.is_available():
    device = 'cuda'


def tensor_to_PIL_image(image):
    reverse_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda t: tensor_to_image_numpy(t)),
        torchvision.transforms.ToPILImage()
    ])

    return reverse_transforms(image)


def make_time_str(total_seconds):
    s = total_seconds % 60
    total_seconds = total_seconds // 60
    min = total_seconds % 60
    total_seconds = total_seconds // 60
    hr = total_seconds % 24
    days = total_seconds // 24

    return f'{days}d {hr}h {min}min {s}s'


def save_not_transformed_tensor_image(tensor, name):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda t: t.squeeze()),
        torchvision.transforms.Lambda(lambda t: t.permute(1, 2, 0)), # permute CHW to HWC
        torchvision.transforms.Lambda(lambda t: t * 255),
        torchvision.transforms.Lambda(lambda t: t.numpy().astype(numpy.uint8)),
        torchvision.transforms.ToPILImage()
    ])

    image = transform(tensor)
    image.save(name)


def show_diffusion_steps(model, training_set_transformed_no_resize):
    global device

    print('diffusion steps')

    image = training_set_transformed_no_resize[0]
    image = image.to(device=device,
                     non_blocking=True)

    number_of_images = 10

    diffusion_steps = []

    divisions = number_of_images - 1
    start = 0
    step = (model.total_timesteps - start) // divisions

    for index in tqdm(range(start, model.total_timesteps, step)):
        timestep_batch = torch.Tensor([index]).to(device=device,
                                                  non_blocking=True,
                                                  dtype=torch.long)
        image_noisy, noise = model.q_sample(image, timestep_batch)

        diffusion_steps.append(tensor_to_PIL_image(image_noisy))

    diffusion_steps = numpy.hstack(diffusion_steps)
    diffusion_steps = Image.fromarray(diffusion_steps)
    diffusion_steps.save('diffusion steps.png')


def show_data(training_set):
    print('raw image')

    # display raw single image

    image = training_set[0]

    save_not_transformed_tensor_image(image, 'raw data.png')

    # display raw image grid

    print('raw image grid')

    rows = 5
    columns = 5
    figure = matplotlib.pyplot.figure(figsize=(9, 9))
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()

    for i in tqdm(range(1, rows * columns + 1)):
        random_index = torch.randint(0, len(training_set), size=tuple([1])).item() # batch size = 1
        image = training_set[random_index]
        axes = figure.add_subplot(rows, columns, i)
        axes.axis(False)
        axes.imshow(image.squeeze().permute(1, 2, 0))

    figure.savefig('raw data grid.png', bbox_inches='tight', pad_inches=0)

    matplotlib.pyplot.close()


def save_image_batch(image_batch, name='sample images.png'):
    transformed_images = []
    for image in image_batch:
        transformed_image = tensor_to_PIL_image(image)
        transformed_images.append(transformed_image)
    # transformed_images = numpy.vstack(transformed_images))
    transformed_images = numpy.hstack(transformed_images)
    transformed_images = Image.fromarray(transformed_images)
    transformed_images.save(name)


def tensor_to_image_numpy(image):
    reverse_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda t: t.cpu().detach().squeeze()),
        torchvision.transforms.Lambda(lambda t: (t + 1) / 2),
        torchvision.transforms.Lambda(lambda t: t.permute(1, 2, 0)), # permute CHW to HWC
        torchvision.transforms.Lambda(lambda t: t * (2**8 - 1)), # scale to 255
        torchvision.transforms.Lambda(lambda t: t.numpy().astype(numpy.uint8))
    ])

    return reverse_transforms(image)


def save_images_to_gif(image_batch_timesteps, name='diffusion.gif', fps=60, dpi=80, image_size=(64, 64)):
    transfered_images = []

    for image in image_batch_timesteps:
        transfered_images.append(tensor_to_image_numpy(image[0]))

    height = image_size[0]
    width = image_size[1]
    figure_size = (width / dpi, height / dpi)
    figure, ax = matplotlib.pyplot.subplots(figsize=figure_size, dpi=dpi)
    ax.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
    images = []

    for image in transfered_images:
        im = ax.imshow(image, animated=True)
        images.append([im])

    animate = matplotlib.animation.ArtistAnimation(figure, images, interval=0, blit=True, repeat=True, repeat_delay=2000)
    animate.save(filename=name, fps=fps)

    matplotlib.pyplot.close()


def load_model_and_sample(train, milestone, training_dataloader):
    global device

    train.load(milestone)

    # print training stats
    print('training time: ' + make_time_str(train.total_training_time))

    train.plot_loss()

    if (len(train.training_loss_values) > 0):
        print(f'training loss: {train.training_loss_values[-1]}')

    # sample the model
    print('sampling image')

    image_size=(128, 128)
    images = train.model.sample(image_size=image_size, batch_size=1, return_all_timesteps=True)
    save_image_batch(images[-1], 'sample images.png')
    save_image_batch(images[0], 'sample images inital.png')

    print('generating gif')
    save_images_to_gif(image_batch_timesteps=images, image_size=image_size)

    # visualize architecture
    print('model architecture')

    batch_x = next(iter(training_dataloader))
    batch_x = batch_x.to(device=device,
                         non_blocking=True)

    timestep = torch.full(size=tuple([1]), # batch size = 1
                          fill_value=train.model.total_timesteps,
                          device=device,
                          dtype=torch.long)

    writer = SummaryWriter('tensorboard/') # python -m tensorboard.main --logdir=tensorboard
    writer.add_graph(train.model.model, (batch_x, timestep))
    writer.close()


def filter_file_paths_not_includes(file_list, string_list):
    filtered = []

    for file_name in file_list:
        contains_none = True

        for string in string_list:
            if string in str(file_name):
                contains_none = False
                break

        if contains_none:
            filtered.append(file_name)

    return filtered


def filter_file_paths_includes(file_list, string_list):
    filtered = []

    for file_name in file_list:
        for string in string_list:
            if string in str(file_name):
                filtered.append(file_name)
                break

    return filtered


def filter_file_paths_includes_all(file_list, string_list):
    filtered = []

    for file_name in file_list:
        contains_all = True

        for string in string_list:
            if not string in str(file_name):
                contains_all = False
                break

        if contains_all:
            filtered.append(file_name)

    return filtered


def main():
    # seed = 314
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    image_transform_no_resize = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), # scales data into [0, 1]
        torchvision.transforms.Lambda(lambda t: t * 2 - 1) # scales data to [-1, 1]
    ])

    print('loading dataset')

    image_dataset_directory = ''

    training_set = ImageDataset(directory=image_dataset_directory,
                                transform=torchvision.transforms.ToTensor())

    training_set_transformed_no_resize = ImageDataset(directory=image_dataset_directory,
                                                    transform=image_transform_no_resize)

    training_set_transformed = ImageDataset(directory=image_dataset_directory)

    # filter out images if they have tags in their file names
    remove_list = [
        'includes none of the elements'
    ]

    includes_all_list = [
        'includes all elements'
    ]

    includes_list = [
        'includes at least 1 element'
    ]

    training_set.image_paths = filter_file_paths_not_includes(training_set.image_paths, remove_list)
    training_set.image_paths = filter_file_paths_includes_all(training_set.image_paths, includes_all_list)
    training_set.image_paths = filter_file_paths_includes(training_set.image_paths, includes_list)

    training_set_transformed_no_resize.image_paths = filter_file_paths_not_includes(training_set_transformed_no_resize.image_paths, remove_list)
    training_set_transformed_no_resize.image_paths = filter_file_paths_includes_all(training_set_transformed_no_resize.image_paths, includes_all_list)
    training_set_transformed_no_resize.image_paths = filter_file_paths_includes(training_set_transformed_no_resize.image_paths, includes_list)

    training_set_transformed.image_paths = filter_file_paths_not_includes(training_set_transformed.image_paths, remove_list)
    training_set_transformed.image_paths = filter_file_paths_includes_all(training_set_transformed.image_paths, includes_all_list)
    training_set_transformed.image_paths = filter_file_paths_includes(training_set_transformed.image_paths, includes_list)

    # make data loader

    batch_size = 2**4

    training_dataloader = DataLoader(dataset=training_set_transformed,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=os.cpu_count(),
                                                pin_memory=True,
                                                drop_last=False)

    print(f'training data batches ({batch_size} images per batch): {len(training_dataloader)}')

    unet = UNet()
    diffusion_model = GaussianDiffusion(model=unet)
    train = Train(diffusion_model=diffusion_model,
                  data_loader=training_dataloader,
                  save_latest_only=True,
                  sample_every=500,
                  save_every=500,
                  training_epochs=100000,
                  results_directory='./results',
                  sample_batch_image_size=(128, 128),
                  use_ddim_sampling=True,
                  learning_rate=8e-5)


    show_data(training_set)
    show_diffusion_steps(diffusion_model, training_set_transformed_no_resize)

    # start training
    train.train()

    # continue training
    # train.load('latest')
    # train.train()

    # sample
    # train.load('latest')
    # train.use_ddim_sampling = False
    # train.sample_batch_size = 16
    # train.sample(name='manual')
    # or
    load_model_and_sample(train, 'latest', training_dataloader)


if __name__ == '__main__':
    main()
