"""Implementation of Kernel Foveated Rendering (KFR) algorithm."""
import torch
import torch.nn.functional as F
import numpy as np


def generate_coordinate_meshgrid(height, width, dtype=torch.float32, device="cpu"):
    """Generate a coordinate meshgrid.

    Args:
        height: Height of the meshgrid.
        width: Width of the meshgrid.
        dtype: Data type of the meshgrid.
        device: Device of the meshgrid.
    """
    coordinate_meshgrid = torch.meshgrid(
        torch.arange(0, width, dtype=dtype, device=device),
        torch.arange(0, height, dtype=dtype, device=device),
        indexing="xy",
    )
    coordinate_meshgrid = torch.stack(coordinate_meshgrid, dim=-1)
    return coordinate_meshgrid


def get_l_value(gaze_position, image_height, image_width):
    """Get the L value for the log-polar buffer.

    Args:
        gaze_position: A tensor of shape (2,) representing the gaze position as (x,y) pixel coordinates.
        image_size: The image size as (width, height) tuple or tensor.

    Returns:
        A float representing the L value.
    """
    l_values = ((gaze_position[0], gaze_position[1]),
                (gaze_position[0], image_height - gaze_position[1]),
                (image_width - gaze_position[0], gaze_position[1]),
                (image_width - gaze_position[0], image_height - gaze_position[1]))
    l_values = torch.tensor(l_values, dtype=torch.float32)
    l_values = torch.norm(l_values, dim=1)
    l_value = torch.log(torch.max(l_values))
    return l_value


def gaussian_blur(image, sigma=1, kernel_size=3, padding_mode="replicate"):
    """Apply Gaussian blur to an image.

    Args:
        image: A tensor of shape (height, width, channels).
        sigma: The standard deviation of the Gaussian kernel.

    Returns:
        A tensor of shape (height, width, channels) representing the blurred image.
    """
    image = image.permute(2, 0, 1).unsqueeze(1)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
    kernel = torch.exp(-torch.arange(-kernel_size // 2 + 1,
                       kernel_size // 2 + 1) ** 2 / (2 * sigma ** 2))
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.view(1, 1, -1)
    kernel = kernel * kernel.transpose(-1, -2)
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.to(image.device)
    padding_amount = (kernel_size//2, kernel_size//2,
                      kernel_size//2, kernel_size//2)
    image = F.conv2d(F.pad(image, padding_amount, padding_mode), kernel)
    return image.squeeze(1).permute(1, 2, 0)


def logpolar_to_cartesian_coords(logpolar_coordinates, gaze_position, alpha,
                                 image_height, image_width):
    dtype = logpolar_coordinates.dtype
    device = logpolar_coordinates.device

    logpolar_height, logpolar_width = logpolar_coordinates.shape[:2]
    logpolar_buffer_resolution = torch.tensor(
        (logpolar_width, logpolar_height), dtype=dtype, device=device)
    logpolar_coordinates = logpolar_coordinates / logpolar_buffer_resolution
    l_value = get_l_value(gaze_position, image_height, image_width)
    u = torch.pow(logpolar_coordinates[..., 0], alpha)
    u = torch.exp(l_value * u)
    v = logpolar_coordinates[..., 1] * 2 * np.pi
    x = u * torch.cos(v) + gaze_position[0]
    y = u * torch.sin(v) + gaze_position[1]
    cartesian_coordinates = torch.stack((x, y), dim=-1)
    return cartesian_coordinates


def cartesian_to_logpolar_coords(cartesian_coordinates, gaze_position, alpha,
                                 logpolar_height, logpolar_width):
    image_height, image_width = cartesian_coordinates.shape[:2]
    logpolar_buffer_coords = cartesian_coordinates - gaze_position
    # Get L value
    l_value = get_l_value(gaze_position, image_height, image_width)
    u = torch.log(torch.norm(logpolar_buffer_coords, dim=-1)) / l_value
    u = torch.clamp(u, min=0)  # Clamp to avoid -inf in the center
    u = torch.pow(u, 1/alpha) * logpolar_width
    v = torch.atan2(logpolar_buffer_coords[..., 1],
                    logpolar_buffer_coords[..., 0])
    v = (v + 2 * np.pi) % (2 * np.pi)
    v = v * logpolar_height / (2 * np.pi)
    return torch.stack((u, v), dim=-1)


def cartesian_to_logpolar_buffer(image, gaze_position, sigma=1.8, alpha=4, use_gaussian_blur=True):
    """Converts an image to a logpolar buffer.

    Args:
        image: A tensor of shape (height, width, channels).
        gaze_position: A tensor of shape (2,) representing the gaze position as (x,y) pixel coordinates.
        sigma: Size of the cartesian buffer to log-polar buffer.
        alpha: Power of the kernel.
        use_gaussian_blur: Apply a 3x3 Gaussian blur to the right side of the image.

    Returns:
        A tensor of shape (height, width, channels).
    """
    image_height, image_width = image.shape[:2]
    logpolar_buffer_height = int(round(image_height / sigma))
    logpolar_buffer_width = int(round(image_width / sigma))
    logpolar_buffer_coords = generate_coordinate_meshgrid(
        logpolar_buffer_height, logpolar_buffer_width,
        dtype=image.dtype, device=image.device)
    cartesian_coords = logpolar_to_cartesian_coords(
        logpolar_buffer_coords, gaze_position, alpha,
        image_height, image_width)
    x = 2 * cartesian_coords[..., 0] / (image_width - 1) - 1
    y = 2 * cartesian_coords[..., 1] / (image_height - 1) - 1
    cartesian_coords = torch.stack((x, y), dim=-1).unsqueeze(0)
    image_channels_first = image.permute(2, 0, 1).unsqueeze(0)
    logpolar_image = F.grid_sample(
        image_channels_first, cartesian_coords,
        align_corners=True, padding_mode="border")
    logpolar_image = logpolar_image.squeeze(0).permute(1, 2, 0)
    if use_gaussian_blur:
        right_side = logpolar_image[:, logpolar_buffer_width // 2:]
        right_side = gaussian_blur(right_side)
        logpolar_image = torch.cat(
            (logpolar_image[:, :logpolar_buffer_width // 2], right_side), dim=1)
    return logpolar_image


def logpolar_to_cartesian_image(logpolar_buffer, gaze_position, sigma=1.8, alpha=4,
                                image_height=None, image_width=None):
    """Converts a logpolar buffer to an image.

    Args:
        logpolar_buffer: A tensor of shape (height, width, channels).
        gaze_position: A tensor of shape (2,) representing the gaze position as (x,y) pixel coordinates.
        sigma: Size of the cartesian buffer to log-polar buffer.
        alpha: Power of the kernel.
        image_height: Height of the output image.
        image_width: Width of the output image.

    Returns:
        A tensor of shape (height, width, channels).
    """
    if not isinstance(gaze_position, torch.Tensor):
        gaze_position = torch.tensor(
            gaze_position, dtype=logpolar_buffer.dtype, device=logpolar_buffer.device)
    if image_height is None or image_width is None:
        image_height = int(round(logpolar_buffer.shape[0] * sigma))
        image_width = int(round(logpolar_buffer.shape[1] * sigma))
    logpolar_height, logpolar_width = logpolar_buffer.shape[:2]
    cartesian_coordinates = generate_coordinate_meshgrid(
        image_height, image_width,
        dtype=logpolar_buffer.dtype, device=logpolar_buffer.device)
    logpolar_coordinates = cartesian_to_logpolar_coords(
        cartesian_coordinates, gaze_position, alpha,
        logpolar_height, logpolar_width)
    u = 2 * logpolar_coordinates[..., 0] / (logpolar_width - 1) - 1
    v = 2 * logpolar_coordinates[..., 1] / logpolar_height - 1
    logpolar_buffer_coords = torch.stack((u, v), dim=-1).unsqueeze(0)

    logpolar_buffer_channels_first = logpolar_buffer.permute(
        2, 0, 1).unsqueeze(0)
    # Add one row to the bottom of the logpolar buffer to allow interpolation around 0/2pi boundary.
    logpolar_buffer_channels_first = torch.cat(
        (logpolar_buffer_channels_first, logpolar_buffer_channels_first[:, :, 0:1, :]), dim=2)
    image = F.grid_sample(
        logpolar_buffer_channels_first, logpolar_buffer_coords,
        align_corners=True, padding_mode="border")
    return image.squeeze(0).permute(1, 2, 0)
