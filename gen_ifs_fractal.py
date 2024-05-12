import os
from pathlib import Path
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from fractal_config import (
    FractalConfig,
    DRAGONS_CURVE,
    SUPER_DRAGONS_CURVE,
    SIERPINSKI_TRIANGLE,
    SUPER_SIERPINSKI_TRIANGLE,
    CURVY_DRAGON_CURVE,
    SIERPINSKI_DRAGON,
    SUPER_SIERPINSKI_DRAGON,
    RANDOM_FRACTAL
)

class IFSFractal(nn.Module):
    def __init__(
        self,
        transformations: List[Tuple[torch.Tensor, torch.Tensor]],
        colors: List[Tuple[float, float, float]],
        probabilities: Optional[List[float]] = None,
        activation: str = 'selu',
        device: Union[str, torch.device] = 'cpu',
        **kwargs
    ):
        """
        Initialize an IFSFractal object.

        :param transformations: A list of tuples containing transformation matrices and translation vectors.
        :param colors: A list of tuples representing RGB colors for each transformation.
        :param probabilities: A list of probabilities for each transformation. If None, equal probabilities are used.
        :param activation: The activation function to use for the transformations.
        :param device: The device to use for computations (e.g., 'cpu' or 'cuda').
        """
        super(IFSFractal, self).__init__()
        self.device = device
        self.dimension = transformations[0][0].shape[0]
        self.transformations = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dimension, self.dimension, bias=True),
                self.get_activation(activation),
            ) for _ in transformations
        ]).to(device)
        for i, (matrix, vector) in enumerate(transformations):
            assert matrix.shape == (
                self.dimension, self.dimension), f"Transformation matrix {i} should be {self.dimension}x{self.dimension}."
            assert vector.shape == (
                self.dimension,), f"Translation vector {i} should be {self.dimension}-dimensional."
            self.transformations[i][0].weight = nn.Parameter(
                matrix, requires_grad=False)
            self.transformations[i][0].bias = nn.Parameter(
                vector, requires_grad=False)
        self.colors = torch.tensor(
            colors,
            device=self.device,
            dtype=torch.float32) / 255.0
        if (probabilities is None) or (len(probabilities) != len(transformations)):
            self.probabilities = torch.ones(
                len(transformations),
                device=self.device,
                dtype=torch.float32) / len(transformations)
        elif isinstance(probabilities, torch.Tensor):
            self.probabilities = probabilities
        else:
            self.probabilities = torch.tensor(
                probabilities, device=self.device, dtype=torch.float32)
            
        # Stack transformation matrices and bias vectors
        self.matrices = torch.stack([trans[0].weight for trans in self.transformations]).cuda()
        self.biases = torch.stack([trans[0].bias for trans in self.transformations]).cuda()

    def get_activation(self, activation: str) -> nn.Module:
        """
        Get the activation function module based on the provided name.

        :param activation: The name of the activation function.
        :return: The corresponding PyTorch activation function module.
        """
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'selu': nn.SELU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'softplus': nn.Softplus(),
            'softsign': nn.Softsign(),
            'hardshrink': nn.Hardshrink(),
            None: nn.Identity(),
            'none': nn.Identity(),
            'identity': nn.Identity(),
        }
        return activations[activation]

    def forward(self,
                points: torch.Tensor,
                prev_colors: torch.Tensor) -> Tuple[torch.Tensor,
                                                    torch.Tensor]:
        """
        Perform the forward pass of the IFSFractal.

        :param points: The input points tensor.
        :param prev_colors: The previous colors tensor.
        :return: The transformed points and updated colors.
        """
        with torch.no_grad():
            choices = torch.multinomial(self.probabilities, points.size(0), replacement=True)
            # Select transformation matrices and bias vectors based on choices
            selected_matrices = self.matrices[choices]
            selected_biases = self.biases[choices]

            # Apply transformations in a single step
            transformed_points = torch.bmm(
                points.unsqueeze(1),
                selected_matrices).squeeze(1) + selected_biases

            # Apply activation function
            transformed_points = self.transformations[0][1](transformed_points)

            # Update colors in a single step
            selected_colors = self.colors[choices]
            new_colors = (prev_colors + selected_colors) / 2

        return transformed_points, new_colors


def rasterize_points(
    points: torch.Tensor,
    colors: torch.Tensor,
    size: int = 2048,
    batch_size: int = 1000000,
    colorspace: str = 'RGB'
) -> Image.Image:
    """
    Rasterize 2D points into an image.

    :param points: The 2D points tensor.
    :param colors: The colors tensor corresponding to the points.
    :param size: The size of the output image.
    :param batch_size: The batch size for processing points.
    :param colorspace: The colorspace gen_ifs_fractal.py FractalConfig.pyof the output image.
    :return: The rasterized image.
    """
    assert points.shape[1] == 2, "The rasterizer only supports 2D points."

    # Normalize points to the range [0, 1]
    min_vals = points.min(0).values
    max_vals = points.max(0).values
    points = (points - min_vals) / (max_vals - min_vals)

    # Create a tensor to hold the color values for each pixel
    color_tensor = torch.zeros(
        (size, size, 3), dtype=torch.float32, device=points.device)

    # Create a count tensor to store the number of points per pixel
    count = torch.zeros((size, size), dtype=torch.float32,
                        device=points.device)

    # Process points in batches
    num_batches = (points.shape[0] + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="Rasterizing"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, points.shape[0])
        batch_points = points[start_idx:end_idx]
        batch_colors = colors[start_idx:end_idx]

        # Scale points to the desired image size
        batch_points = (batch_points * (size - 1)).round().long()

        # Create a mask for valid points within the image bounds
        mask = (batch_points[:, 0] >= 0) & (batch_points[:, 0] < size) & (
            batch_points[:, 1] >= 0) & (batch_points[:, 1] < size)
        batch_points = batch_points[mask]
        batch_colors = batch_colors[mask]

        # Accumulate color values and point counts
        color_tensor.index_put_(
            (batch_points[:, 0], batch_points[:, 1]), batch_colors, accumulate=True)
        count.index_put_((batch_points[:, 0], batch_points[:, 1]), torch.ones_like(
            batch_points[:, 0], dtype=torch.float32), accumulate=True)

    # Compute the average color for each pixel
    color_tensor = color_tensor / count.unsqueeze(-1)

    # Handle pixels with no points
    color_tensor[count == 0] = 0.0

    # Convert the color tensor to uint8
    image_tensor = (color_tensor * 255).byte()

    return Image.fromarray(image_tensor.cpu().numpy(), colorspace)


def generate_fractal(config: dict, device: Union[str, torch.device] = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a fractal based on the provided configuration.

    :param config: The configuration dictionary for the fractal.
    :param device: The device to use for computations (e.g., 'cpu' or 'cuda').
    :return: The generated points and colors tensors.
    """
    ifs_fractal = IFSFractal(**config, device=device).to(device)
    dimension = ifs_fractal.dimension
    points = torch.zeros((config['batch_size'], config['num_iterations'] + 1, dimension), device=device)
    colors = torch.zeros((config['batch_size'], config['num_iterations'] + 1, 3), device=device)
    points[:, 0, :] = torch.zeros(dimension, device=device)
    colors[:, 0, :] = torch.rand((config['batch_size'], 3), device=device)

    for i in tqdm(range(1, config['num_iterations'] + 1), desc="Generating Fractal"):
        points[:, i, :], colors[:, i, :] = ifs_fractal(points[:, i - 1, :], colors[:, i - 1, :])

    return points.view(-1, dimension), colors.view(-1, 3)


def save_image(image: Image.Image, filename: str):
    """
    Save the image to a file with a unique filename.

    :param image: The image to save.
    :param filename: The base filename.
    """
    folder, name = os.path.split(filename)
    name, extension = os.path.splitext(name)

    counter = 1
    file_path = Path(folder, f"{name}{extension}")

    # If the file exists, find a new filename
    while file_path.exists():
        file_path = Path(folder, f"{name}_{counter}{extension}")
        counter += 1

    image.convert("RGB").save(file_path)
    print(f"File saved as: {file_path}")

def generate_fractal_image(config: dict, device: Union[str, torch.device] = 'cpu', size: int = 2048, batch_size: int = 1000000) -> Image.Image:
    print(f"Generating {config.name}...")
    points, colors = generate_fractal(config.__dict__, device)
    return rasterize_points(points, colors, size=size, batch_size=batch_size, colorspace=config.colorspace)

