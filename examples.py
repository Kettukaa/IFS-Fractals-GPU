import torch
from gen_ifs_fractal import  save_image, generate_fractal_image
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



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    configs = [
        DRAGONS_CURVE,
        SIERPINSKI_TRIANGLE,
        CURVY_DRAGON_CURVE,
        SIERPINSKI_DRAGON,
        RANDOM_FRACTAL(),
        RANDOM_FRACTAL(),
        RANDOM_FRACTAL(),
    ]
    
    for config in configs:
        image = generate_fractal_image(config, device=device, size=8192, batch_size=1000000)
        save_image(image, f"{config.name}.png")
