import base64
import gc
import json
import os

import numpy as np
import numpy.typing as npt
from rembg import remove, new_session
import torch

# Using a fork of pygltflib
# https://github.com/samaust/pygltflib refactor branch
from pygltflib.v2.schema import (
    Accessor,
    Accessor_componentType,
    Attributes,
    Buffer,
    BufferView,
    BufferView_target,
    DATA_URI_HEADER,
    Material,
    Material_alphaMode,
    MaterialPbrMetallicRoughness,
    Mesh,
    MeshPrimitive,
    Mesh_primitive_mode,
    Node,
    Scene,
)
from pygltflib import GLTF2

from .depthcrafter.inference import DepthCrafterImage
from .depthcrafter.utils import save_video, vis_sequence_depth


def remove_bg(
    images,
    #frame_rate: int,
    #save_remove_background: bool,
    #save_remove_background_dir: str,
    frame_load_cap: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Remove background and replace by solid color

    Returns :
    - tuple(output_video_path, out_frames__float32)
    """
    images_cpu = images.cpu()
    # print("images_cpu.shape = {}".format(images_cpu.shape))

    # Iterate over images
    frames_uint8 = []  # To save as video
    frames_float32 = []  # To send to step 2
    masks_float32 = []  # To send to step3
    for i, frame in enumerate(images_cpu):
        if i == frame_load_cap:
            break

        frame_uint8 = (frame.numpy() * 255).astype(np.uint8)
        frame_uint8_ndarray = np.array(frame_uint8)

        # Remove background and replace by bgcolor
        session = new_session("u2net_human_seg")

        mask_float32 = np.array(remove(
            data=frame_uint8_ndarray,
            session=session,
            only_mask=True)).astype(np.float32) / 255.0
        masks_float32.append(mask_float32)
        frame_float32 = frame_uint8_ndarray.astype(np.float32) / 255.0

        # Append to output video
        frames_uint8.append(frame_uint8_ndarray)
        frames_float32.append(frame_float32)

    # Save video
    # remove_background_video_path = ""
    # if save_remove_background:
    #     output_video_path = os.path.join(save_remove_background_dir, "rembg.mp4")  # TODO generate unique path
    #     remove_background_video_path = save_video(
    #         video_frames=frames_uint8,
    #         output_video_path=output_video_path,
    #         fps=frame_rate,
    #         crf=18,
    #         source_dtype=np.uint8
    #     )

    # print("len(frames_float32) = {}".format(len(frames_float32)))

    # Stack numpy list of arrays
    out_video_float32 = np.stack(frames_float32)
    # print("type(out_video_float32) = {}".format(type(out_video_float32)))
    # print("out_video_float32.shape = {}".format(out_video_float32.shape))

    out_masks_float32 = np.stack(masks_float32)

    return (out_video_float32,
            out_masks_float32,)


def compute_frames_normalized_disparities(
        frames: npt.NDArray[np.float32],
        unet_path: str = "tencent/DepthCrafter",
        pre_train_path: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        cpu_offload: str = "model",
        num_inference_steps: int = 5,
        guidance_scale: float = 1.0,
        window_size: int = 110,
        overlap: int = 25,
        seed: int = 42,
        track_time: bool = False
):
    """
    Compute frames normalized disparities
    """
    depthcrafter_demo = DepthCrafterImage(
        unet_path=unet_path,
        pre_train_path=pre_train_path,
        cpu_offload=cpu_offload,
    )
    # Infer normalized disparities
    ndisps = depthcrafter_demo.infer(
        frames,
        num_inference_steps,
        guidance_scale,
        window_size=window_size,
        overlap=overlap,
        seed=seed,
        track_time=track_time
    )
    # frames_type = type(frames)
    # frames_shape = frames.shape
    # print(f"type(frames) : {frames_type}")
    # print(f"frames.shape : {frames_shape}")

    # ndisps_type = type(ndisps)
    # ndisps_shape = ndisps.shape
    # print(f"type(ndisps) : {ndisps_type}")
    # print(f"ndisps.shape : {ndisps_shape}")

    # Release memory
    del depthcrafter_demo
    gc.collect()
    torch.cuda.empty_cache()

    return ndisps


def compute_ndisps(
    in_video_array: npt.NDArray[np.float32],
    frame_rate: int,
    cfd_num_inference_steps: int,
    cfd_guidance_scale: float,
    cfd_window_size: int,
    cfd_overlap: int,
    cfd_seed: int,
    save_ndisps: bool,
    save_ndisps_dir: str,
) -> tuple[str, npt.NDArray[np.float32]]:
    """
    Process images and return normalized disparities
    """
    # if in_video_array is None:
    #     print("TridiGeneratorPcdEstimation compute_ndisps in_video_array is None")
    #     return None

    frames = in_video_array
    # print("in_array.shape = {}".format(in_video_array.shape))
    # print("frames.shape = {}".format(frames.shape))

    out_ndisps_float32 = compute_frames_normalized_disparities(
        frames=frames,
        num_inference_steps=cfd_num_inference_steps,
        guidance_scale=cfd_guidance_scale,
        window_size=cfd_window_size,
        overlap=cfd_overlap,
        seed=cfd_seed
    )
    # print("out_ndisps_float32.shape = {}".format(out_ndisps_float32.shape))
    # print("type(out_ndisps_float32) = {}".format(type(out_ndisps_float32)))

    # assert ndisps datatype
    assert type(out_ndisps_float32) is np.ndarray

    # visualize the depth map and save the results
    ndisps_video = vis_sequence_depth(out_ndisps_float32)

    # Save ndisps video
    out_ndisps_video_path = ""
    if save_ndisps:
        output_video_path = os.path.join(save_ndisps_dir, "ndisps.mp4")  # TODO generate unique path
        out_ndisps_video_path = save_video(
            video_frames=ndisps_video,
            output_video_path=output_video_path,
            fps=frame_rate,
            crf=18,
            source_dtype=np.float32
        )

    return (out_ndisps_video_path, out_ndisps_float32,)


def compute_pcd(frames, ndisps, masks, fovz_deg, zmin, zmax):
    """
    Compute a pointcloud from a video array, a ndisps array, a masks array,
    a horizontal fields of view, a min depth and a max depth.

    NOTE : This algorithm seems to work but it might contain errors.
    """
    num_frames = ndisps.shape[0]
    H = ndisps.shape[1]
    W = ndisps.shape[2]

    fovx = fovz_deg*np.pi/180.0
    fovy = fovx * H / W
    tx = np.linspace(-fovx/2.0, fovx/2.0, W)
    ty = np.linspace(-fovy/2.0, fovy/2.0, H)

    A = zmax*0.1
    B = A/zmin - 0.1

    masks_expanded = np.expand_dims(masks, axis=-1)
    frames_alpha = np.concatenate((frames, masks_expanded), axis=-1)

    positions = []
    colors = []
    for i in range(num_frames):
        position_image = np.where(np.zeros([H, W]) == 0)
        v = np.array(position_image[0])
        u = np.array(position_image[1])
        d = ndisps[i, v, u]

        zc = -A/(B*d+0.1)
        xc = -zc * np.tan(tx[u-1])
        yc = zc * np.tan(ty[v-1])

        positions.append(np.stack((xc, yc, zc), axis=1))
        colors.append(frames_alpha[i, v, u])

    out_positions = np.stack(positions)
    out_colors = np.stack(colors)

    return (out_positions, out_colors,)


def apply_gamma_correction(image: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    # print("image.shape : {}".format(image.shape))

    # Normalize image to [0,1], apply gamma correction, then scale back to [0,255]
    RGB = image[:, :3]
    A = image[:, 3]
    A_expanded = np.expand_dims(A, axis=-1)
    RGB_corrected = np.power(RGB / 255.0, gamma) * 255.0
    RGBA_corrected = np.concatenate((RGB_corrected, A_expanded), axis=-1).astype(np.uint8)

    return RGBA_corrected


def get_uri_from_blob(blob):
    data = base64.b64encode(blob).decode('utf-8')
    uri = f'{DATA_URI_HEADER}{data}'

    return uri


def create_gltf(positions, colors) -> GLTF2:
    """
    Saves a point cloud with vertex colors to a GLTF file.

    Parameters:
    - in_positions (numpy ndarray): ndarray of points positions.
    - in_colors (numpy ndarray): ndarray of points colors
    - save_dir (str) : absolute path to save directory
    """
    # print("points.shape : {}".format(positions.shape))
    # print("colors.shape : {}".format(colors.shape))

    # Type conversions
    frames_positions = positions.astype(np.float32)  # Convert to float32
    frames_colors_original = (colors * 255).astype(np.uint8)  # Convert to uint8 [0, 255]
    frames_colors = []
    # TODO : edit apply_gamma_correction to allow to work without for loop
    for frame_colors_original in frames_colors_original:
        frames_colors.append(apply_gamma_correction(frame_colors_original, gamma=2.0))
    frames_colors = np.stack(frames_colors)

    # Get binary blobs data
    positions_binary_blob = frames_positions.tobytes()
    colors_binary_blob = frames_colors.tobytes()

    # Caculate buffers byteLength
    positions_buffer_byteLength = len(positions_binary_blob)
    colors_buffer_byteLength = len(colors_binary_blob)

    # Create buffers to store point and color data
    position_buffer = Buffer(
        uri=get_uri_from_blob(positions_binary_blob),
        byteLength=positions_buffer_byteLength
    )  # 4 bytes per float32
    color_buffer = Buffer(
        uri=get_uri_from_blob(colors_binary_blob),
        byteLength=colors_buffer_byteLength
    )  # 1 byte per uint8

    scenes = []
    bufferViews = []
    accessors = []
    meshes = []
    nodes = []
    for i, (frame_positions, frame_colors) in enumerate(zip(frames_positions, frames_colors)):
        # Calculate positions count and byteLength
        positions_count = len(frame_positions)
        positions_bufferView_byteLength = len(frame_positions.flatten()) * 4  # np.float32

        # Calculate colors count
        colors_count = len(frame_colors)
        colors_bufferView_byteLength = len(frame_colors.flatten())  # np.uint8

        # Create buffer views
        point_buffer_view = BufferView(
            buffer=0,
            byteOffset=i*positions_bufferView_byteLength,
            byteLength=positions_bufferView_byteLength,
            target=BufferView_target.ARRAY_BUFFER.value
        )
        color_buffer_view = BufferView(
            buffer=1,
            byteOffset=i*colors_bufferView_byteLength,
            byteLength=colors_bufferView_byteLength,
            target=BufferView_target.ARRAY_BUFFER.value
        )
        bufferViews.append(point_buffer_view)
        bufferViews.append(color_buffer_view)

        # Create accessor for point cloud positions
        positions_max = np.max(frame_positions, 0).tolist()
        positions_min = np.min(frame_positions, 0).tolist()
        position_accessor = Accessor(
            bufferView=i*2,
            byteOffset=0,
            componentType=Accessor_componentType.FLOAT.value,
            count=positions_count,
            type="VEC3",
            max=positions_max,
            min=positions_min
        )
        # Create accessor for point cloud colors
        color_accessor = Accessor(
            bufferView=i*2+1,
            byteOffset=0,
            componentType=Accessor_componentType.UNSIGNED_BYTE.value,
            normalized=True,
            count=colors_count,
            type="VEC4"  # RGBA
        )
        accessors.append(position_accessor)
        accessors.append(color_accessor)

        # Create mesh primitives for points and colors
        attributes = Attributes(
            POSITION=i*2,
            COLOR_0=i*2+1
        )
        primitive = MeshPrimitive(
            attributes=attributes,
            indices=None,
            mode=Mesh_primitive_mode.POINTS.value,
            material=0
        )
        # Create the mesh object
        meshes.append(Mesh(
            name="Object_{}".format(i),
            primitives=[primitive]
        ))

        # Create the node object
        nodes.append(Node(mesh=i))

        # Create the scene
        scenes.append(Scene(nodes=[i]))

    # Create a simple material using the vertex colors
    material = Material(
        name="vertexColorMaterial",
        doubleSided=None,
        pbrMetallicRoughness=MaterialPbrMetallicRoughness(
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            metallicFactor=None,
            roughnessFactor=None
        ),
        alphaMode=Material_alphaMode.MASK.value,
        alphaCutoff=0.5,
        emissiveFactor=None,
        extensions={"KHR_materials_unlit": {}},  # Optional: Use unlit material (no lighting)
    )
    extensionsUsed = ["KHR_materials_unlit"]

    # Create the GLTF object
    gltf = GLTF2()

    # Add everything to the GLTF object
    gltf.extensionsUsed = extensionsUsed
    gltf.buffers = [position_buffer, color_buffer]
    gltf.bufferViews = bufferViews
    gltf.accessors = accessors
    gltf.materials = [material]  # Add the unlit material
    gltf.meshes = meshes
    gltf.nodes = nodes
    gltf.scenes = scenes
    gltf.scene = 0

    return gltf


def save_gltf(gltf, save_output_dir: str,) -> str:
    """
    Saves a glTF object containing a point cloud with vertex colors to a GLTF file.

    Parameters:
    - in_positions (numpy ndarray): ndarray of points positions.
    - in_colors (numpy ndarray): ndarray of points colors
    - save_output_dir (str) : absolute path to save directory
    """
    gltf_file_path = os.path.join(save_output_dir, "points.gltf")
    # print(f"gltf_file_path : {gltf_file_path}")

    # Save gltf
    gltf.save(gltf_file_path)

    return gltf_file_path


def pretty_print_gltf_json(gltf) -> str:
    """
    Pretty print glTF json
    """
    gltf_JSON = gltf.gltf_to_json()
    json_object = json.loads(gltf_JSON)
    json_formatted_str = json.dumps(json_object, indent=2)

    return json_formatted_str
