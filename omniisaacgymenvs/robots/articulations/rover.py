# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import Optional
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from utils.terrain_utils.terrain_generation import *
import carb

#from omniisaacgymenvs.utils.terrain_utils.terrain_utils import *

class Rover(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "Rover",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name
        #prim_path="/"
        if self._usd_path is None:
            #assets_root_path = get_assets_root_path()
            assets_root_path = "http://localhost:8080/omniverse://127.0.0.1/"
            #assets_root_path = "http://100.127.177.125:8080/omniverse://100.127.177.125"
            # print(assets_root_path)
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            #self._usd_path ="/home/anton/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/cartpole.usd"#assets_root_path + "/Projects/usd/exomy/exomy_model/cartpole.usd" #"http://localhost:8080/omniverse://100.127.177.125/Projects/usd/exomy/exomy_model/Cartpole.usd"#"./cartpole.usd"#assets_root_path + "/Isaac/Robots/Cartpole/cartpole.usd"#/home/anton/OmniIsaacGymEnvs/omniisaacgymenvs/robots/articulations/cartpole.usd"#assets_root_path + "/Isaac/Robots/Cartpole/cartpole.usd"
            #self._usd_path = assets_root_path + "/Projects/usd/exomy/exomy_model/exomy_model3.usd"
            #self._usd_path = assets_root_path + "/Projects/exomy/exomy_model3.usd"
            #self._usd_path = "http://localhost:8080/omniverse://127.0.0.1/Projects/exomy/Mars_Rover_2_COPY6.usd"
            #self._usd_path = "http://localhost:8080/omniverse://127.0.0.1/Projects/RoverS/test.usd"
            #self._usd_path = "http://localhost:8080/omniverse://127.0.0.1/Projects/exomy/exomy_model.usd"
            #self._usd_path = "http://localhost:8080/omniverse://127.0.0.1/Projects/exomy/mars_rover_2_working.usd"
            self._usd_path = "http://localhost:8080/omniverse://127.0.0.1/Projects/simplified5.usd"
            #self._usd_path = "robots/articulations/simplified3.usd"

        add_reference_to_stage(self._usd_path, prim_path)   
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

