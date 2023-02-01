import os

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

import kfr


def main():
    image_filename = "image.png"
    output_dir = "outputs"
    dtype = torch.float32
    device = torch.device("cuda")
    fovea_patch_size = 128
    gaze_position = (498, 163)

    os.makedirs(output_dir, exist_ok=True)
    image = Image.open(image_filename).convert("RGB")
    image = torch.tensor(np.array(image), dtype=dtype, device=device) / 255.0
    reference_image_filename = os.path.join(output_dir, "reference_image.png")
    plt.imsave(reference_image_filename, image.cpu().numpy())

    reference_fovea_crop_filename = os.path.join(
        output_dir, "reference_fovea_crop.png")
    reference_fovea_crop = image[
        gaze_position[1] - fovea_patch_size // 2:gaze_position[1] + fovea_patch_size // 2,
        gaze_position[0] - fovea_patch_size // 2:gaze_position[0] + fovea_patch_size // 2]
    plt.imsave(reference_fovea_crop_filename,
               reference_fovea_crop.cpu().numpy())

    lp_buffer = kfr.cartesian_to_logpolar_buffer(
        image, gaze_position, True)
    lp_buffer_filename = os.path.join(output_dir, "lp_buffer.png")
    plt.imsave(lp_buffer_filename, lp_buffer.clamp(0, 1).cpu().numpy())

    foveated_image = kfr.logpolar_to_cartesian_image(
        lp_buffer, gaze_position, image_height=image.shape[0], image_width=image.shape[1])
    foveated_image_filename = os.path.join(output_dir, "foveated_image.png")
    plt.imsave(foveated_image_filename, foveated_image.cpu().numpy())

    foveated_fovea_crop_filename = os.path.join(
        output_dir, "foveated_fovea_crop.png")
    foveated_fovea_crop = foveated_image[
        gaze_position[1] - fovea_patch_size // 2:gaze_position[1] + fovea_patch_size // 2,
        gaze_position[0] - fovea_patch_size // 2:gaze_position[0] + fovea_patch_size // 2]
    plt.imsave(foveated_fovea_crop_filename,
               foveated_fovea_crop.cpu().numpy())


if __name__ == "__main__":
    main()
