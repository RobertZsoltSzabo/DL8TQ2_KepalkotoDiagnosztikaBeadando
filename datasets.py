from torch.utils.data import Dataset
from torchvision.transforms import Grayscale
from pathlib import Path
from PIL import Image

class GrayscaleImageFolder(Dataset):
    def __init__(self, root, image_extension='png', transform=None):
        self.transform = transform
        self.image_files = list(Path(root).glob(f'**/*.{image_extension}'))
        self.to_grayscale = Grayscale(1)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        path = self.image_files[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
            img = self.to_grayscale(img)
        return img
    

class GrayscaleImageFolderWithFilenameFilter(Dataset):
    def __init__(self, root, filename_filter, image_extension='png', transform=None):
        self.transform = transform
        self.image_files = list(Path(root).glob(f'**/*{filename_filter}*.{image_extension}'))
        self.to_grayscale = Grayscale(1)

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        path = self.image_files[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
            img = self.to_grayscale(img)
        return img