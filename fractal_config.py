from typing import List, Tuple, Optional, Union
import math
import torch
import random

class FractalConfig:
    def __init__(
        self,
        name: str,
        transformations: List[Tuple[torch.Tensor, torch.Tensor]],
        colors: List[Tuple[float, float, float]],
        colorspace: str = 'HSV',
        probability_weights: Optional[List[float]] = None,
        activation: Optional[str] = None,
        batch_size: int = 1000,
        num_iterations: int = 100000
    ):
        self.name = name
        self.transformations = transformations
        self.colors = colors
        self.colorspace = colorspace
        self.probabilities = (probability_weights)/sum(probability_weights) if probability_weights is not None else None
        self.activation = activation
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    @property
    def dimension(self) -> int:
        return self.transformations[0][0].shape[0]

    @property
    def num_transformations(self) -> int:
        return len(self.transformations)

    def __repr__(self) -> str:
        return f"FractalConfig(name='{self.name}', dimension={self.dimension}, num_transformations={self.num_transformations})"
    
    def fractal_transpose(self) -> 'FractalConfig':
        transformations = []
        for matrix, vector in self.transformations:
            transformations.append((matrix.T, -matrix.T @ vector))
        return FractalConfig(
            name=self.name,
            transformations=transformations,
            colors=self.colors,
            colorspace=self.colorspace,
            probability_weights=self.probabilities,
            activation=self.activation,
            batch_size=self.batch_size,
            num_iterations=self.num_iterations
        )
    
    def fractal_inverse(self) -> 'FractalConfig':
        transformations = []
        for matrix, vector in self.transformations:
            transformations.append((torch.inverse(matrix), -torch.inverse(matrix) @ vector))
        return FractalConfig(
            name=self.name,
            transformations=transformations,
            colors=self.colors,
            colorspace=self.colorspace,
            probability_weights=self.probabilities,
            activation=self.activation,
            batch_size=self.batch_size,
            num_iterations=self.num_iterations
        )
    
    def print_config(self):
        print(f"FractalConfig(name='{self.name}', dimension={self.dimension}, num_transformations={self.num_transformations})")
        print(f"Transformations: {self.transformations}")
        print(f"Colors: {self.colors}")
        print(f"Colorspace: {self.colorspace}")
        print(f"Probability Weights: {self.probabilities}")
        print(f"Activation: {self.activation}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Number of Iterations: {self.num_iterations}")
    
    def increase_dimension(self, target_dimension: int) -> 'FractalConfig':
        if self.dimension >= target_dimension:
            return self

        transformations = []
        colors = self.colors.copy()

        for matrix, vector in self.transformations:
            new_matrix = torch.eye(target_dimension, dtype=matrix.dtype)
            new_matrix[:self.dimension, :self.dimension] = matrix
            new_vector = torch.zeros(target_dimension, dtype=vector.dtype)
            new_vector[:self.dimension] = vector
            transformations.append((new_matrix, new_vector))

        for _ in range(target_dimension - self.dimension):
            colors.append(tuple(random.randint(0, 255) for _ in range(3)))

        return FractalConfig(
            name=self.name,
            transformations=transformations,
            colors=colors,
            colorspace=self.colorspace,
            probability_weights=self.probabilities,
            activation=self.activation,
            batch_size=self.batch_size,
            num_iterations=self.num_iterations
        )

    @classmethod
    def compose(cls, *configs: 'FractalConfig', name: Optional[str] = None) -> 'FractalConfig':
        if not configs:
            raise ValueError("At least one configuration must be provided for composition.")

        if name is None:
            name = "_".join(config.name for config in configs)

        max_dimension = max(config.dimension for config in configs)
        configs = [config.increase_dimension(max_dimension) for config in configs]

        transformations = []
        colors = []
        probabilities = []

        for config in configs:
            transformations.extend(config.transformations)
            colors.extend(config.colors)
            if config.probabilities is not None:
                probabilities.extend(config.probabilities)

        if not probabilities:
            probabilities = None
        else:
            total_prob = sum(probabilities)
            probabilities = [prob / total_prob for prob in probabilities]

        return cls(
            name=name,
            transformations=transformations,
            colors=colors,
            colorspace=configs[0].colorspace,
            probability_weights=probabilities,
            activation=configs[0].activation,
            batch_size=configs[0].batch_size,
            num_iterations=configs[0].num_iterations
        )
        
        
# Define fractal configurations
DRAGONS_CURVE = FractalConfig(
    name="DragonsCurve",
    transformations=(
        (torch.tensor([
            [(1 / math.sqrt(2)) * math.cos(math.pi / 4), -(1 / math.sqrt(2)) * math.sin(math.pi / 4)],
            [(1 / math.sqrt(2)) * math.sin(math.pi / 4),  (1 / math.sqrt(2)) * math.cos(math.pi / 4)]
        ]), torch.tensor([0.0, 0.0])),
        (torch.tensor([
            [-(1 / math.sqrt(2)) * math.cos(3 * math.pi / 4), -(1 / math.sqrt(2)) * math.sin(3 * math.pi / 4)],
            [ (1 / math.sqrt(2)) * math.sin(3 * math.pi / 4), -(1 / math.sqrt(2)) * math.cos(3 * math.pi / 4)]
        ]), torch.tensor([1.0, 0.0]))
    ),
    colors=(
        (0, 127.5, 205),
        (255, 127.5, 100),
    ),
)

SUPER_DRAGONS_CURVE = FractalConfig(
    name="SuperDragonsCurve",
    transformations=(
        (torch.tensor([
            [(1 / math.sqrt(2)) * math.cos(math.pi / 4), -(1 / math.sqrt(2)) * math.sin(math.pi / 4)],
            [(1 / math.sqrt(2)) * math.sin(math.pi / 4),  (1 / math.sqrt(2)) * math.cos(math.pi / 4)]
        ]), torch.tensor([0.0, 0.0])),
        (torch.tensor([
            [-(1 / math.sqrt(2)) * math.cos(3 * math.pi / 4), -(1 / math.sqrt(2)) * math.sin(3 * math.pi / 4)],
            [ (1 / math.sqrt(2)) * math.sin(3 * math.pi / 4), -(1 / math.sqrt(2)) * math.cos(3 * math.pi / 4)]
        ]), torch.tensor([1.0, 0.0])),
        (torch.tensor([
            [(1 / math.sqrt(2)) * math.cos(math.pi / 4), -(1 / math.sqrt(2)) * math.sin(math.pi / 4)],
            [(1 / math.sqrt(2)) * math.sin(math.pi / 4),  (1 / math.sqrt(2)) * math.cos(math.pi / 4)]
        ]), torch.tensor([0.0, 1.0])),
        (torch.tensor([
            [-(1 / math.sqrt(2)) * math.cos(3 * math.pi / 4), -(1 / math.sqrt(2)) * math.sin(3 * math.pi / 4)],
            [ (1 / math.sqrt(2)) * math.sin(3 * math.pi / 4), -(1 / math.sqrt(2)) * math.cos(3 * math.pi / 4)]
        ]), torch.tensor([1.0, 1.0]))
    ),
    colors=(
        (0, 127.5, 205),
        (255, 127.5, 100),
        (255, 255, 0),
        (255, 127.5, 100),
    ),
    num_iterations=1000000,
)

SIERPINSKI_TRIANGLE = FractalConfig(
    name="SierpinskiTriangle",
    transformations=(
        (torch.tensor([
            [0.5, 0.0],
            [0.0, 0.5]
        ]), torch.tensor([0.0, 0.0])),
        (torch.tensor([
            [0.5, 0.0],
            [0.0, 0.5]
        ]), torch.tensor([0.5, 0.0])),
        (torch.tensor([
            [0.5, 0.0],
            [0.0, 0.5]
        ]), torch.tensor([0.25, 0.5]))
    ),
    colors=(
        (0, 127.5, 205),
        (255, 127.5, 10),
        (180, 190, 254)
    ),
    num_iterations=200000,
)

SUPER_SIERPINSKI_TRIANGLE = FractalConfig(
    name="SuperSierpinskiTriangle",
    transformations=(
        (torch.tensor([
            [0.5, 0.0],
            [0.0, 0.5]
        ]), torch.tensor([0.0, 0.0])),
        (torch.tensor([
            [0.5, 0.0],
            [0.0, 0.5]
        ]), torch.tensor([0.5, 0.0])),
        (torch.tensor([
            [0.5, 0.0],
            [0.0, 0.5]
        ]), torch.tensor([0.25, 0.5])),
        (torch.tensor([
            [0.5, 0.0],
            [0.0, 0.5]
        ]), torch.tensor([0.125, 0.25])),
        (torch.tensor([
            [0.5, 0.0],
            [0.0, 0.5]
        ]), torch.tensor([0.375, 0.25])),
        (torch.tensor([
            [0.5, 0.0],
            [0.0, 0.5]
        ]), torch.tensor([0.25, 0.75])),
    ),
    colors=(
        (0, 127.5, 205),
        (255, 127.5, 10),
        (180, 190, 254),
        (0, 127.5, 205),
        (255, 127.5, 10),
        (180, 190, 254),
    ),
    activation='selu',
    num_iterations=200000,
)

CURVY_DRAGON_CURVE = FractalConfig(
    name="CurvyDragonCurve",
    transformations=DRAGONS_CURVE.transformations,
    colors=DRAGONS_CURVE.colors,
    activation='selu',
)

SIERPINSKI_DRAGON = FractalConfig(
    name="SierpinskiDragon",
    transformations=(
        *DRAGONS_CURVE.transformations,
        *SIERPINSKI_TRIANGLE.transformations
    ),
    colors=(
        *DRAGONS_CURVE.colors,
        (0, 127.5, 205),
        (0, 127.5, 205),
        (0, 127.5, 205),
    ),
    probability_weights=torch.tensor([1, 1, 6, 6, 6]),
    activation='selu',
)

SUPER_SIERPINSKI_DRAGON = FractalConfig(
    name="SuperSierpinskiDragon",
    transformations=(
        *SIERPINSKI_DRAGON.transformations,
        *SUPER_SIERPINSKI_TRIANGLE.transformations
    ),
    colors=(
        *SIERPINSKI_DRAGON.colors,
        *SUPER_SIERPINSKI_TRIANGLE.colors
    ),
    probability_weights=torch.tensor([1., 1., 8., 8., 8., 1., 1., 8.]),
    activation='selu',
)

randomAngle = 2*math.pi*random.random()
randomScale = random.random()
RANDOM_FRACTAL = lambda :  FractalConfig(
    name="RandomFractal",
    transformations=[
        (torch.tensor([
            [randomScale*math.cos(randomAngle), -randomScale*math.sin(randomAngle)],
            [randomScale*math.sin(randomAngle),  randomScale*math.cos(randomAngle)]
        ]), torch.tensor([random.random(), -random.random()]),
        ), (torch.tensor([
            [-randomScale*math.cos(randomAngle), -randomScale*math.sin(randomAngle)],
            [randomScale*math.sin(randomAngle),  -randomScale*math.cos(randomAngle)]
        ]),  torch.tensor([-random.random(), random.random()]))
    ],
    colors=[
        (0, 127.5, 205),
        (255, 127.5, 100),
    ],
    probability_weights=torch.randint(1, 10, (2,)).float(),
)
