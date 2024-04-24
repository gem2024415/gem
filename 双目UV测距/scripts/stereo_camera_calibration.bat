set image_dir=data\lenacv-camera
set save_dir=configs\lenacv-camera
set width=8
set height=11
set square_size=20 
set image_format=jpg
::show_2dimage=True
set show=False
::left camera calibration
python mono_camera_calibration.py ^
    --image_dir  %image_dir% ^
    --image_format %image_format%  ^
    --square_size %square_size%  ^
    --width %width%  ^
    --height %height%  ^
    --prefix left  ^
    --save_dir %save_dir% ^
    --show %show%

::right camera calibration
python mono_camera_calibration.py ^
    --image_dir  %image_dir% ^
    --image_format  %image_format%  ^
    --square_size %square_size%  ^
    --width %width%  ^
    --height %height%  ^
    --prefix right  ^
    --save_dir %save_dir% ^
    --show %show%

::stereo camera calibration
python stereo_camera_calibration.py ^
    --left_file %save_dir%/left_cam.yml ^
    --right_file %save_dir%/right_cam.yml ^
    --left_prefix left ^
    --right_prefix right ^
    --width %width% ^
    --height %height% ^
    --left_dir %image_dir% ^
    --right_dir %image_dir% ^
    --image_format  %image_format%  ^
    --square_size %square_size% ^
    --save_dir %save_dir% ^
