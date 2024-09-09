from typing import Optional
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

class RoverView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "RoverView"
    ) -> None:
        """[summary]
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
            reset_xform_properties=False
        )
        self._actuated_dof_indices = list()
        self._actuated_pos_indices = list()
        self._actuated_vel_indices = list()
        self._num_pos_dof = 6
        self._num_vel_dof = 6

    @property
    def actuated_dof_indices(self):
        return self._actuated_dof_indices
    @property
    def actuated_pos_indices(self):
        return self._actuated_pos_indices

    

    @property
    def actuated_vel_indices(self):
        return self._actuated_vel_indices

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)
        self._actuated_dof_indices = [i for i in range(self.num_dof)]
        # self._actuated_vel_indices = [9,10,11,12,13,14] exomy
        # self._actuated_pos_indices = [3,4,5,6,7,8] exomy
        # self._actuated_vel_indices = [10, 11, 14, 7, 8, 13] #[FR, CR, RR, FL, CL, RL] # WITH DIFFERENTIAL
        # self._actuated_pos_indices = [5, 12, 3, 9]#[FR ,RR, FL, RL] # WITH DIFFERENTIAL
        self._actuated_vel_indices = [10, 5, 12, 9, 3, 11] #[FR, CR, RR, FL, CL, RL] 
        self._actuated_pos_indices = [6, 8, 4, 7]#[FR ,RR, FL, RL]
        self._passive_pos_indices = [6, 8, 4, 7]#[FR ,RR, FL, RL]
        print("Initializing123")
        print(self._actuated_dof_indices)
        # self.physics_rotors = [RigidPrimView(prim_paths_expr=f"/World/envs/.*/Ingenuity/rotor_physics_{i}", name=f"physics_rotor_{i}_view", reset_xform_properties=False) for i in range(2)]
        # self.visual_rotors = [RigidPrimView(prim_paths_expr=f"/World/envs/.*/Ingenuity/rotor_visual_{i}", name=f"visual_rotor_{i}_view", reset_xform_properties=False) for i in range(2)]