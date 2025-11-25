import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

class MuJoCoPendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path("pendulum.xml")
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = np.random.uniform(-np.pi, np.pi)
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[0] = action[0]
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = - (self.data.qpos[0]**2 + 0.1 * self.data.qvel[0]**2 + 0.001 * action[0]**2)
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        theta = self.data.qpos[0]
        theta_dot = self.data.qvel[0]
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

if __name__ == "__main__":
    env = MuJoCoPendulumEnv(render_mode="human")
    obs, _ = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
