# ComfyUI-TridiGenerator
ComfyUI nodes to generate 3D from videos and render in the browser/AR/VR

## Objectives

Generating :

* Convert videos to colored point clouds images sequences
* (low priority) Convert videos to stereoscopic side by side images sequences

Viewing :

* Look at the 3D points clouds in the browser
* Watch the 3D colored point cloud videos in AR/VR
* Watch the side by side images videos in AR/VR

## Current nodes

### TridiGeneratorPcdEstimation

A ComfyUI node to estimate a point cloud from images, disparity, masks,
fovz in degrees, zmin in meters and zmax in meters.

The node was tested to work with an input video of 55 frames and a cfd_window_size of 55 frames on a video card with 24GB of VRAM.

## Future nodes

### TridiGeneratorPcdMaskBackground

Compute a mask that allows to hide the points in the background during rendering.

### TridiGeneratorPcdComputeNDisps

Compute normalized disparity.

### TridiGeneratorPcdComputePCD

Compute point cloud.

### TridiGeneratorPcdSaveGLTF

Create and save a glTF file. Only works with a low number of frames.

### TridiGeneratorPcdSaveOpenEXR

Write layers and channels to OpenEXR file :

* colors (R, G, B, A) at full resolution
* position (P.X, P.Y, P.Z) at full or lower resolution
* (optionnally) normalised disparity ND.V at full or lower resolution

### TridiGeneratorPcdLoadOpenEXR

Read OpenEXR file saved by TridiGeneratorPcdSaveOpenEXR.

### TridiGeneratorPcdPreviewOpenEXR

Display a single frame or multiple frames of an OpenEXR file loaded by TridiGeneratorPcdLoadOpenEXR.


## Future improvements

The current implementation of TridiGeneratorPcdEstimation does many calculations in a single node. A future version will split the node in many nodes.

The current implementation of TridiGeneratorPcdEstimation saves the results in a glTF file. This only works for a small amount of frames because the files become too large and fail to load in Three.js library.

Table 1 : glTF file (json and binary data in the same file)

| Number of frames | Filesize | Status |
| ---------------- | -------- | ------ |
| 2                | 42 MB    | Works  |
| 10               | 213 MB   | Works  |
| 20               | 426 MB   | Works  |
| 40               | 853 MB   | Fails to load in Three.js |

A future version will replace glTF by OpenImageIO OpenEXR. It supports layers, multiple channels, compression, HALF, FLOAT, and UINT32 pixel data. It could allow to store :

* COLOR : R, G, B, A
* POSITION : P.X, P.Y, P.Z
* Normalized disparity : ND.V

Once rendering works, it could be interesting to experiment with interpolation of the point positions between frames.

## Viewing the colored point cloud videos stored in a glTF file

The current Three.js app is not shared online.

I generated a colored point cloud that contains one point per pixel. Each frame is stored in a different scene.

 I tested a simple Three.js app that loads the glTF file and changes the visibility of the scenes over time so that it looks like a moving colored point cloud.

CONS

* This method only works with a small amount of frames because of issues related to loading large glTF files in the browser.
* It requires to store X,Y,Z coordinates for each point.
* The performance of the app is bad.

## Viewing the colored point cloud videos generated from a OpenEXR images sequence

WIP. Will share when I have a working solution.

OpenEXR PROS :

* Supported by good quality libraries in multiple programming languages.
* Can store sequences of images.
* Can compress and decompress the data (could reduce filesize by half).
* Possible to store colors (R, G, B, A) in a lower resolution data type
* Possible to store position (P.X, P.Y, P.Z) in the same resolution as the colors or a lower resolution.
* Possible to store normalized disparity value (ND.V). Normalized disparity is used to compute position (P.X, P.Y, P.Z).

Work planned :

* Split TridiGeneratorPcdEstimation in multiple nodes and output colors (R, G, B, A) and positions (P.X, P.Y, P.Z) data.
* Develop a node that takes colors (R, G, B, A) and positions (P.X, P.Y, P.Z) as input and writes an OpenEXR images sequence file.
* Develop a Three.js app that read the colors (R, G, B, A) and positions (P.X, P.Y, P.Z) images sequences and render colored points clouds.

A few implementations will be tested :

Experiment 1 - During loading, create points sequences using the colors and positions and store them in memory.

Experiment 2 - Load the colors and positions sequences as if they were videos. Pass these videos to a shader that will calculate points positions in realtime.

Experiment 3 - Load the colors and disparity sequences as if they were videos. Pass these videos to a shader that will calculate points positions in realtime. Heavier load than option 2 but allows to experiment with 2D to 3D conversion parameters in realtime.

Experiment 4 - Instead of creating points for each frame of changing their visibility, create points and change their positions.

The first experiment might improve rendering performance as the points positions will be pre-calculated.

The second experiment might make it possible and easier to do position interpolation. It might also be possible to use a low resolution positions images sequence in a shader to improve performance at the cost of lower position precision.

## Converting videos to stereoscopic side by side images sequences

PROS
No gaps between the pixels

CONS
Cannot move the camera position and orientation
One eye shows the original pixels while the other eye shows generated pixels (might look blurry or wrong).

I experimented a little with [StereoCrafter](https://github.com/TencentARC/StereoCrafter) and prefer the point cloud generation solution. While it might show gaps when too close or when being too far from the initial camera position, it allows to move around in the scene and render it in augmented reality (AR).

## Disclaimer

I don't have much experience using Three.js. It might take a long time to get a working solution.

I'm interested in other ideas. I want to use open source solutions.

The first version of TridiGeneratorPcdEstimation works for a video of resolution 1024x1024. Other resolutions might generate errors.

Future versions might lower memory usage and improve performance.

DepthCrafter might have been improved since work began on this plugin.
TODO : look at commits, issues, pull requests of DepthCrafter to see if they require changes in our code.

## Acknowledgements

* [DepthCrafter](https://github.com/Tencent/DepthCrafter): Generating Consistent Long Depth Sequences for Open-world Videos
* DepthCrafter uses stable_video_diffusion (SVD)
