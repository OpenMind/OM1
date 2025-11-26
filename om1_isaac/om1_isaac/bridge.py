from isaacgym import gymapi
import os

class OM1IsaacGymBridge:
    def __init__(self):
        self.gym = gymapi.acquire_gym()
        self.sim = None
        self.env = None
        self.actor = None

    def create_sim(self):
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0, 0, -9.81)
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = True
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    def load_ant(self):
        asset_root = os.path.expanduser("~/isaacgym/python/isaacgym/assets")
        asset_file = "urdf/ant/ant.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        # create_env richiede: sim, lower, upper, num_per_row
        lower = gymapi.Vec3(-2, -2, 0)
        upper = gymapi.Vec3(2, 2, 3)
        self.env = self.gym.create_env(self.sim, lower, upper, 1)
        pose = gymapi.Transform(gymapi.Vec3(0, 0, 0.5))
        self.actor = self.gym.create_actor(self.env, asset, pose, "ant", 0, 0)

    def step(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

    def get_lidar(self):
        return [1.0, 1.2, 0.8, 1.5, 2.0, 1.1, 0.9, 1.3]
