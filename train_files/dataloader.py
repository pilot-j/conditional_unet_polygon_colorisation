import json
from torch.utils.data import Dataset
from PIL import Image

class PolygonDataset(Dataset):
    def __init__(self, base_path, colour_map, split='training', transform=None):
        self.base_path = base_path
        self.split = split
        self.transform = transform

        with open(f"{base_path}/{split}/data.json") as f:
            self.data = json.load(f)

        # Normalized colour channels - RGB
        self.color_map = colour_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load input image (grayscale polygon)
        input_img = Image.open(f"{self.base_path}/{self.split}/inputs/{item['input_polygon']}")
        input_img = input_img.convert('L')

        # Load output image (colored polygon)
        output_img = Image.open(f"{self.base_path}/{self.split}/outputs/{item['output_image']}")
        output_img = output_img.convert('RGB')

        # Get color tensor
        colour = self.color_map[item['colour']]

        if self.transform:
            input_img = self.transform(input_img)
            output_img = self.transform(output_img)

        return {
            'input': input_img,
            'colour': colour.float(),
            'output': output_img
        }
