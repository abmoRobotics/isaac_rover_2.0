## Old repository

https://github.com/abmoRobotics/isaac_rover

# General
This repository contains a reinforcement learning implementation in Isaac Sim 2022.1.1 for learning to navigate in an unstructured Mars environment. The implementation is based on a custom built rover platform (based on the design of M2020), made by Group 750 at Aalborg University (Autumn 2022).

**Design**
![10](https://user-images.githubusercontent.com/56405924/200193226-f0ae8f5f-2c59-45ce-a091-d0b832dbc6ac.JPG)
**Internals**
![12](https://user-images.githubusercontent.com/56405924/200193231-ff1713ef-f4f9-46c4-8d7d-28ef6c3dc83d.JPG)
**Reinforcement Learning**

https://user-images.githubusercontent.com/56405924/204378992-13709e09-ca9a-4aa7-b76d-eb012e801e02.mp4

https://user-images.githubusercontent.com/56405924/204389953-d40bbcf7-2219-49d9-8480-07725625d674.mp4

**Terrain**


https://user-images.githubusercontent.com/56405924/212775394-6bf902e0-1161-42c6-aa53-8a0d6f10f324.mp4



https://user-images.githubusercontent.com/56405924/212775747-afaa91c2-4a10-458d-8b66-d3fbc30e07b0.mp4




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
