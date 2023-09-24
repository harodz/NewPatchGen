# NewPatchGen
This repo generates a robust patch for stop signs detected with YoloV5 and test it in carla.

## Usage
1. Install Carla build version of 0.9.14. Need to access the unreal engine to add the stop sign and change the texture using API.
2. Open Carla, use the unreal engine to add the assets in the '/Uasset' folder.
   - To replicate, the stop sign location and roation are set as:
     - Position: 3940,14640,300
     - Rot: 0,0,90
     - ID: BP_Apartment04_v05_Opt1_2
3. Generate a patch using 'main.py'. Outputs are in '/outputs/$date$/$time$/Patch'. Use 'texture.png' in Carla in the following steps
   - Recommend doing it without Carla opened
   - Use full path when redirecting files due to the usage of 'hydra' python package 
4. Run '/Carla_Data_Saver/carla_data_saver.py' to test the patches in Carla.

All configuration can be changed using the yaml files in '/configs' or '/Carla_Data_Saver/configs' so that each session are automatically named and recorded with the change.
