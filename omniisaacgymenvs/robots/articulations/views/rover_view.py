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

        # self.physics_rotors = [RigidPrimView(prim_paths_expr=f"/World/envs/.*/Ingenuity/rotor_physics_{i}", name=f"physics_rotor_{i}_view", reset_xform_properties=False) for i in range(2)]
        # self.visual_rotors = [RigidPrimView(prim_paths_expr=f"/World/envs/.*/Ingenuity/rotor_visual_{i}", name=f"visual_rotor_{i}_view", reset_xform_properties=False) for i in range(2)]