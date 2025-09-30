import os

import torch

import folder_paths

from ..utils import (
    remove_bg,
    compute_ndisps,
    compute_pcd,
    create_gltf,
    save_gltf,
    pretty_print_gltf_json
)


class TridiGeneratorPcdEstimation:
    """
    A ComfyUI node to estimate a point cloud from images, disparity, masks,
    fovz in degrees, zmin in meters and zmax in meters.
    """
    @classmethod
    def INPUT_TYPES(s):
        """
        Defines the input types for the node.
        """
        return {
            "required": {
                # Data
                "images": ("IMAGE",),
                # General parameters
                "frame_rate": ("INT", {"default": 16, "min": 1, "step": 1}),
                "remove_background": ("BOOLEAN", {"default": True}),
                "frame_load_cap": ("INT", {"default": 110, "min": 1, "step": 1, "tooltip": "Maximum number of frames to load"}),
                # compute_frames_disparities parameters
                "cfd_num_inference_steps": ("INT", {"default": 5, "min": 1, "step": 1}),
                "cfd_guidance_scale": ("FLOAT", {"default": 1.0, "min": 0, "max": 100, "step": 0.001}),
                "cfd_window_size": ("INT", {"default": 110, "min": 1, "step": 1}),
                "cfd_overlap": ("INT", {"default": 25, "min": 1, "step": 1}),
                "cfd_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                # Pcd estimation parameters
                "fov_deg": ("INT", {"default": 74, "min": 1, "max": 120, "step": 1, "tooltip": "Field of view, horizontal (degree)"}),
                "min_depth_m": ("FLOAT", {"default": 2.0, "min": 0, "max": 100, "step": 0.001, "tooltip": "Min depth from camera (m)"}),
                "max_depth_m": ("FLOAT", {"default": 3.0, "min": 0, "max": 100, "step": 0.001, "tooltip": "Max depth from camera (m)"}),
                # Output parameters
                "filename_prefix": ("STRING", {"default": "TridiGenerator"}),
                "format": (["gltf"],),
                #"save_remove_background": ("BOOLEAN", {"default": False}),
                "save_ndisps": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
                "print_gltf_json": ("BOOLEAN", {"default": False}),
                # Render parameters
                "xr_mode": (["VR", "AR"],),  # Dropdown for XR mode selection
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        # "rembg_video_path",
        "ndisp_video_path",
        "glTF_file_path",
        "glTF_json",
    )
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "TridiGenerator"
    DESCRIPTION = "Estimate a point cloud video from images."

    def run(
        self,
        images=None,
        frame_rate: int = 16,
        remove_background: bool = True,
        frame_load_cap: int = 110,
        cfd_num_inference_steps: int = 5,
        cfd_guidance_scale: float = 1,
        cfd_window_size: int = 110,
        cfd_overlap: int = 25,
        cfd_seed: int = 42,
        fov_deg: int = 74,
        min_depth_m: float = 2.0,
        max_depth_m: float = 3.0,
        filename_prefix: str = "TridiGenerator",
        format: str = "gltf",
        #save_remove_background: bool = False,
        save_ndisps: bool = False,
        save_output: bool = True,
        print_gltf_json: bool = False,
        xr_mode: str = "VR"
    ):
        if images is None:
            return {
                "ui": {
                    "format": [format],
                    "output_url": [""],
                },
                "result": ("","","",)
            }

        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return {
                "ui": {
                    "format": [format],
                    "output_url": [""],
                },
                "result": ("","","",)
            }

        # save_remove_background_dir = (
        #     folder_paths.get_output_directory()
        #     if save_remove_background
        #     else folder_paths.get_temp_directory()
        # )

        save_ndisps_dir = (
            folder_paths.get_output_directory()
            if save_ndisps
            else folder_paths.get_temp_directory()
        )

        save_output_dir = (
            folder_paths.get_output_directory()
            if save_output
            else folder_paths.get_temp_directory()
        )

        # Remove background (calculate mask that will be saved as color alpha)
        # TODO: make optional
        # if remove_background:
        # TODO: optimize, a previous version did remove the backdground
        #       the current version does some conversions en images and pass them out
        #       this can be optimized by doing it outside for loop
        rembg_video_float32, masks_float32 = remove_bg(
            images,
            # frame_rate,
            # save_remove_background,
            # save_remove_background_dir,
            frame_load_cap
        )

        # Generate disparity and normalized disparity [0, 1]
        ndisps_video_path, ndisps_array = compute_ndisps(
            rembg_video_float32,
            frame_rate,
            cfd_num_inference_steps,
            cfd_guidance_scale,
            cfd_window_size,
            cfd_overlap,
            cfd_seed,
            save_ndisps,
            save_ndisps_dir,
        )

        # Compute point cloud
        positions, colors = compute_pcd(
            rembg_video_float32,
            ndisps_array,
            masks_float32,
            fov_deg,
            min_depth_m,
            max_depth_m
        )

        # Create glTF
        gltf = create_gltf(positions, colors)

        # Save point cloud to glTF file
        output_file_path = ""
        if save_output:
            output_file_path = save_gltf(
                gltf,
                save_output_dir
            )

        # Get the glTF filename to use is an url by the frontend
        output_file_url = os.path.basename(output_file_path)

        # Pretty print glTF json
        gltf_json = ""
        if print_gltf_json:
            gltf_json = pretty_print_gltf_json(gltf)

        return {
            "ui": {
                "format": [format],
                "output_url": [output_file_url],
            },
            "result": (
                # rembg_video_path,
                ndisps_video_path,
                output_file_path,
                gltf_json,
            )
        }
