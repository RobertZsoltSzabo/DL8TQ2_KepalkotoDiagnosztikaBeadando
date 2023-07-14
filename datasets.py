from torch.utils.data import Dataset
from torchvision.transforms import Grayscale
from pathlib import Path
from PIL import Image
    

class GrayscaleImageFolder(Dataset):
    '''
    Custom dataset based on the concept of the built-in ImageFolder dataset. Expects single channel (grayscale) inputs.
    args:
        - root: root folder where the images are stored
        - filename_filter: only files that have the provided string in their title will be fetched
        - image_extension: file extension of images
        - transform: transforms applied to returned data at fetch time
    '''
    def __init__(self, root, filename_filter='', image_extension='png', transform=None):
        self.transform = transform
        self.image_files = [file for file in list(Path(root).glob(f'**/*.{image_extension}')) if filename_filter in str(file)]
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