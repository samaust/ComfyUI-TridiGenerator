from diffusers.training_utils import set_seed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import numpy as np
import numpy.typing as npt
import torch

from .depth_crafter_ppl import DepthCrafterPipeline
from .unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter  # noqa: E501


class DepthCrafterImage:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = "model",
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(  # noqa: E501
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # load weights of other components from the provided checkpoint
        self.pipe: DepthCrafterPipeline | DiffusionPipeline = DepthCrafterPipeline.from_pretrained(  # noqa: E501
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # for saving memory, we can offload the model to CPU,
        # or even run the model sequentially to save more memory
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will slow, but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to("cuda")
        # enable attention slicing and xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
            print("Xformers is not enabled")
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        frames: npt.NDArray[np.float32],
        num_denoising_steps: int,
        guidance_scale: float,
        window_size: int = 110,
        overlap: int = 25,
        seed: int = 42,
        track_time: bool = True,
    ):
        # Set seed
        set_seed(seed)

        # print("infer frames.shape = {}".format(frames.shape))

        # inference the depth map using the DepthCrafter pipeline
        with torch.inference_mode():
            disps = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=track_time,
            ).frames[0]  # type: ignore
        # convert the three-channel output to a single channel depth map

        # original implementation
        # disps = disps.sum(-1) / disps.shape[-1]

        # this might improve performance
        disps = disps.mean(-1)


        # print("disps contains nan : ")
        # print(np.isnan(disps).any())

        # normalize the depth map to [0, 1] across the whole video
        disps_min = disps.min()
        ndisps = (disps - disps_min) / (disps.max() - disps_min)

        return ndisps
