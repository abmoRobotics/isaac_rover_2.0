## [New repository](https://github.com/abmoRobotics/RLRoverLab)
This repository is no longer maintained, please refer to our new rover simulations for reinforcement learning based on [Isaac Lab](https://isaac-sim.github.io/IsaacLab/). 


# ***<p style="text-align:center;">[New Rover Simulations Link](https://github.com/abmoRobotics/RLRoverLab) </p>*** 

# General
This repository contains a reinforcement learning implementation in Isaac Sim 2022.1.1 for learning to navigate in an unstructured Mars environment. The implementation is based on a custom built rover platform (based on the design of M2020), made by Group 750 at Aalborg University (Autumn 2022).

**Design**
![10](https://user-images.githubusercontent.com/56405924/200193226-f0ae8f5f-2c59-45ce-a091-d0b832dbc6ac.JPG)
**Internals**
![12](https://user-images.githubusercontent.com/56405924/200193231-ff1713ef-f4f9-46c4-8d7d-28ef6c3dc83d.JPG)
**Reinforcement Learning**

https://github.com/user-attachments/assets/93237c35-daeb-4198-b371-11d1ec10da5e

https://github.com/user-attachments/assets/dfd2dd97-ff95-4df3-a4bb-f6144375ec08

**Terrain**

https://github.com/user-attachments/assets/22ded8e6-47b7-4fcd-9151-81e6b76d2aa1

https://github.com/user-attachments/assets/d30ff35d-84cd-48bb-868c-2aa387d5c8fa

# Setup
1. Install [Isaac Sim](https://developer.nvidia.com/isaac-sim)
2. Clone this repository 
``` bash
git clone https://github.com/abmoRobotics/isaac_rover_2.0.git
cd isaac_rover_2.0
```
3. Set Isaac Sim `PYTHON_PATH`
``` bash
For Linux: alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
For Windows: doskey PYTHON_PATH=C:\Users\user\AppData\Local\ov\pkg\isaac_sim-*\python.bat $*
```
4. Install `isaac_rover_2.0` as a python module for `PYTHON_PATH`:
```bash
PYTHON_PATH -m pip install -e .
```

5. Download terrains from Google Drive:
   1. Download from Google Drive: https://drive.google.com/file/d/1JdcrzYlEAyHXa7hxdZvqeOx7u6Im-JJ6/view?usp=sharing
   2. Unzip files in root folder of the git repository


### Running the training
*Note: All commands should be executed from `omniisaacgymenvs/omniisaacgymenvs`.*

To train a policy, run:

```bash
1. cd omniisaacgymenvs
2. PYTHON_PATH train.py
```

### Terrain generation

Inside the folder terrain_generation you can find a blender project which can be used to create terrains for the simulation. Open the script section and execute the script to get a terrain with a different stone distribution.

When a terrain has been generated, all big rocks are automatically selected, export this as the big_rock_layer that is needed for the collision detection inside the simulation. Then select the whole terrain (all stones and the base terrain) and export is as well. We import the files as ply in IsaacSim, if there are problems exporting into ply, export into fbx and then use a converter.

Apart from the terrain there is also the stone_info file that you need to put into the terrain folder of the simulation. This is used for the spawn and goal validation.
# Contact
For other questions feel free to contact:
* Anton Bjørndahl Mortensen: antonbm2008@gmail.com
