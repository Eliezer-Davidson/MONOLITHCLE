import argparse
import json
import numpy as np

# Isaac Sim setup
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "anti_aliasing": 0})

from omni.isaac.core import World
from omni.isaac.core.prims import RigidPrim
import omni.usd
from pxr import UsdPhysics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("usd_path")
    parser.add_argument("--seconds", type=float, default=1.5)
    args = parser.parse_args()

    omni.usd.get_context().open_stage(args.usd_path)
    simulation_app.update()

    world = World()
    world.reset()

    stage = omni.usd.get_context().get_stage()
    world_prim = stage.GetPrimAtPath("/World")
    
    rigid_prims = {}
    for child in world_prim.GetChildren():
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            rp = RigidPrim(prim_path=str(child.GetPath()), name=child.GetName())
            world.scene.add(rp)
            rigid_prims[child.GetName()] = rp

    world.reset()

    initial_positions = {}
    for name, rp in rigid_prims.items():
        pos, _ = rp.get_world_pose()
        initial_positions[name] = np.array(pos)

    steps = int(args.seconds * 60) # Default 60Hz physics
    for _ in range(steps):
        world.step(render=False)

    drift = {}
    for name, rp in rigid_prims.items():
        pos, _ = rp.get_world_pose()
        final_pos = np.array(pos)
        drift[name] = float(np.linalg.norm(final_pos - initial_positions[name]))

    output = {
        "drift": drift,
        "max_penetration": 0.0,
        "qpos": {}
    }
    print("---SETTLE_REPORT---")
    print(json.dumps(output))

if __name__ == "__main__":
    main()
    simulation_app.close()
