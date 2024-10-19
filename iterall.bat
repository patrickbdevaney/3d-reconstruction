@echo off
REM Directory containing the generated images
set "image_dir=C:\Users\Patrick Devaney\master\generated_images"

REM Path to the run.py script and configuration file
set "run_script_path=.\run.py"
set "config_file=configs\instant-mesh-large.yaml"

REM Output directory
set "output_path=C:\Users\Patrick Devaney\master\generated_meshes"

REM Loop over all images in the directory
for %%i in ("%image_dir%\*.png") do (
    echo Processing %%i...
    python "%run_script_path%" ^
           "%config_file%" ^
           "%%i" ^
           --output_path "%output_path%" ^
           --diffusion_steps 250 ^
           --seed 42 ^
           --scale 1.0 ^
           --distance 4.5 ^
           --view 6 ^
           --export_texmap ^
           --save_video
    echo Finished processing %%i
)

echo All images processed.
