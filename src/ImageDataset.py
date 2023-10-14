from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torchvision


# remove pixel limit
Image.MAX_IMAGE_PIXELS = None


class ImageDataset(Dataset):
    def __init__(
            self,
            directory='',
            training_image_size=(128, 128),
            split=None,
            transform=None,
            data_split_percent=0.8,
            colors='RGB',
            recursive=True
        ):
        super().__init__()

        file_extensions = ('jpg', 'jpeg', 'png', 'webp')
        self.image_paths = []

        pattern = '*.'
        if recursive:
            pattern = '**/*.'

        for file_extension in file_extensions:
            paths = Path(directory).glob(f'{pattern}{file_extension}')
            self.image_paths.extend(paths)

        data_split_index = int(len(self.image_paths) * data_split_percent)
        if split == 'test':
            self.image_paths = self.image_paths[data_split_index:]
        elif split == 'train':
            self.image_paths = self.image_paths[:data_split_index]

        self.transform = transform
        if self.transform is None:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=training_image_size), # height, width
                torchvision.transforms.RandomChoice([
                    torchvision.transforms.RandomHorizontalFlip(p=0.5),
                    # torchvision.transforms.RandomVerticalFlip(p=0.5)
                ]),
                torchvision.transforms.ToTensor(), # scales data into [0, 1]
                torchvision.transforms.Lambda(self.normalize) # scale data to [-1, 1]
            ])

        self.colors = colors


    def normalize(self, x):
        return x * 2 - 1


    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(path).convert(self.colors)
        image = self.transform(image)

        return image
