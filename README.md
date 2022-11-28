## Old repository

https://github.com/abmoRobotics/isaac_rover

# General
This repository contains a reinforcement learning implementation in Isaac Sim 2022.1.1 for learning to navigate in an unstructured Mars environment. The implementation is based on a custom built rover platform (based on the design of M2020), made by Group 750 at Aalborg University (Autumn 2022).

**Design**
![10](https://user-images.githubusercontent.com/56405924/200193226-f0ae8f5f-2c59-45ce-a091-d0b832dbc6ac.JPG)
**Internals**
![12](https://user-images.githubusercontent.com/56405924/200193231-ff1713ef-f4f9-46c4-8d7d-28ef6c3dc83d.JPG)
**Reinforcement Learning**
![13](https://user-images.githubusercontent.com/56405924/204378992-13709e09-ca9a-4aa7-b76d-eb012e801e02.mp4)




# Setup
1. Install [Isaac Sim 2022.1.1](https://developer.nvidia.com/isaac-sim)
2. Clone this repository 
``` bash
git clone https://github.com/abmoRobotics/isaac_rover_2.0.git
cd Pointfilter
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
PYTHON_PATH train.py
```
