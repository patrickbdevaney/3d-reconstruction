# InstantMesh 3D Model Generation: README

## Overview

This project automates the process of generating 3D meshes from images using **InstantMesh**. The provided batch script (`.bat`) is designed to iterate through a folder of images and run `run.py`, generating a 3D model for each image based on predefined configurations. This setup is ideal for large-scale image-to-3D conversion workflows.

In a Linux environment, a shell script should be used instead of the batch file, and directory paths should be modified based on the system's partitioning (root or home directories).

## Prerequisites

Before running the script, make sure you have InstantMesh installed. Follow the instructions in the official InstantMesh repository to build the tool and set up any required dependencies.

### Adjusting Path Settings

Ensure the paths in the batch or shell script point to the correct directories on your system. The script assumes the images are stored in a directory called `generated_images`, and the outputs will be saved in a folder called `generated_meshes`.

If you're on **Windows**, you can modify the provided `.bat` script. For **Linux**, use a similar shell script (example below), adjusting the paths accordingly:

- **Windows** (use absolute paths in the `.bat` file):

    ```batch
    set "image_dir=C:\path\to\your\generated_images"
    set "run_script_path=.\path\to\run.py"
    set "config_file=.\path\to\configs\instant-mesh-large.yaml"
    set "output_path=C:\path\to\your\generated_meshes"
    ```

- **Linux**, you can use a shell script like this:

    ```bash
    #!/bin/bash
    image_dir="/home/user/generated_images"
    run_script_path="./run.py"
    config_file="configs/instant-mesh-large.yaml"
    output_path="/home/user/generated_meshes"
    
    for img in "$image_dir"/*.png; do
        echo "Processing $img..."
        python3 "$run_script_path" \
                "$config_file" \
                "$img" \
                --output_path "$output_path" \
                --diffusion_steps 250 \
                --seed 42 \
                --scale 1.0 \
                --distance 4.5 \
                --view 6 \
                --export_texmap \
                --save_video
        echo "Finished processing $img"
    done
    ```

Modify the paths to suit your directory structure.

### Running the Script

- **Windows**: Double-click the batch file to process all images in the specified directory.
- **Linux**: Make the shell script executable and run it:

    ```bash
    chmod +x run_script.sh
    ./run_script.sh
    ```

## Configurations

The `run.py` script uses parameters like `--diffusion_steps`, `--scale`, `--distance`, etc., which can be adjusted based on your needs. Here’s a brief overview:

- `--output_path`: Specifies the directory to save the generated 3D models.
- `--diffusion_steps`: The number of diffusion steps for generating the mesh.
- `--seed`: Random seed to ensure consistent outputs.
- `--scale`: Adjusts the size of the generated mesh.
- `--distance`: Defines the camera distance in the 3D space.
- `--view`: Specifies the number of views for rendering.
- `--export_texmap`: Exports a texture map for the model.
- `--save_video`: If enabled, a video of the 3D mesh generation is saved.
