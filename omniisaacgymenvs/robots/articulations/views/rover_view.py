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
        self._actuated_vel_indices = [2,4,7,9,12,14]
        self._actuated_pos_indices = [1,3,6,8,11,13]
        print("Initializing123")
        print(self._actuated_dof_indices)
        # self.physics_rotors = [RigidPrimView(prim_paths_expr=f"/World/envs/.*/Ingenuity/rotor_physics_{i}", name=f"physics_rotor_{i}_view", reset_xform_properties=False) for i in range(2)]
        # self.visual_rotors = [RigidPrimView(prim_paths_expr=f"/World/envs/.*/Ingenuity/rotor_visual_{i}", name=f"visual_rotor_{i}_view", reset_xform_properties=False) for i in range(2)]