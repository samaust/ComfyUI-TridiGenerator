from PIL.Image import Image
import tempfile
from typing import Union, List, cast

import matplotlib as mpl
import mediapy
import numpy as np
import numpy.typing as npt
import torch


def save_video(
    video_frames: Union[List[np.ndarray], List[Image]],
    output_video_path: str,
    fps: int = 10,
    crf: int = 18,
    source_dtype: np.uint8 | np.float32 = np.uint8
) -> str:
    if isinstance(video_frames[0], np.ndarray):
        # target dtype is alway np.unint8
        if source_dtype == np.uint8:
            video_frames = [(frame).astype(np.uint8) for frame in
                            cast(npt.NDArray[np.uint8], video_frames)]
        elif source_dtype == np.float32:
            video_frames = [(frame * 255).astype(np.uint8) for frame in
                            cast(npt.NDArray[np.float32], video_frames)]

    elif isinstance(video_frames[0], Image):
        video_frames = [np.array(frame) for frame in video_frames]

    # print("save_video output_video_path type = {}, value = {}".format(type(output_video_path), output_video_path))
    mediapy.write_video(output_video_path, video_frames, fps=fps, crf=crf)
    return output_video_path


class ColorMapper:
    """
    A color mapper to map depth values to a certain colormap
    """
    def __init__(self, colormap: str = "inferno"):
        cmap = mpl.colormaps.get_cmap(colormap)
        cmap_colors_rgba = cmap(np.linspace(0, 1, 256))
        cmap_colors_rgb = cmap_colors_rgba[:, :3] # Keep only R, G, B
        self.cmap_colors = torch.tensor(cmap_colors_rgb, dtype=torch.float32)

    def apply(self, image: torch.Tensor):
        if image.device != self.cmap_colors.device:
            self.cmap_colors = self.cmap_colors.to(image.device)

        indices = torch.clamp(image * 255, 0, 255).long()
        image_colormapped = self.cmap_colors[indices]

        return image_colormapped


def vis_sequence_depth(depths: np.ndarray, v_min=None, v_max=None):
    """
    An util function to apply a colormap to a depths array.
    """
    visualizer = ColorMapper()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_tensor = torch.tensor(depths, dtype=torch.float32).to(device)
    res_tensor = visualizer.apply(depth_tensor)
    res = res_tensor.cpu().numpy()

    return res
