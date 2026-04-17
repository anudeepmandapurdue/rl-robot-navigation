import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time


class NavigationEnv(gym.Env):
    def __init__(self, render=False):
        super().__init__()

        self.render = render

        self.client = p.connect(p.GUI if render else p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Action: [forward, turn]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation: direction + heading
        self.observation_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.robot = None
        self.goal = None
        self.steps = 0

    # =========================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("r2d2.urdf", [0, 0, 0.1])

        self.steps = 0

        self.np_random = np.random.default_rng(seed)
        self.goal = self.np_random.uniform(-2, 2, size=2)

        # visual marker
        p.loadURDF(
            "sphere_small.urdf",
            [self.goal[0], self.goal[1], 0.1],
            globalScaling=2.0
        )

        return self._get_obs(), {}

    # =========================
    def step(self, action):
        self.steps += 1

        action = np.clip(action, -1, 1)
        forward, turn = action

        # smoother scaling (IMPORTANT)
        forward_speed = forward * 2.0
        turn_speed = turn * 1.5

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        rot = p.getMatrixFromQuaternion(orn)

        forward_dir = np.array([rot[0], rot[3], rot[6]])
        velocity = forward_dir * forward_speed

        # IMPORTANT: stabilize rotation (prevents drifting spiral)
        p.resetBaseVelocity(
            self.robot,
            linearVelocity=velocity,
            angularVelocity=[0, 0, turn_speed]
        )

        p.stepSimulation()

        if self.render:
            time.sleep(1 / 240)

        obs = self._get_obs()
        reward = self._compute_reward()

        terminated = self._check_done()
        truncated = self.steps >= 300

        return obs, reward, terminated, truncated, {}

    # =========================
    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        xy = np.array(pos[:2])

        yaw = p.getEulerFromQuaternion(orn)[2]

        # direction to goal (normalized)
        direction = self.goal - xy
        direction = np.clip(direction / 4.0, -1.0, 1.0)

        return np.array([
            direction[0],
            direction[1],
            np.tanh(yaw)
        ], dtype=np.float32)

    # =========================
    def _compute_reward(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        xy = np.array(pos[:2])

        dist = np.linalg.norm(xy - self.goal)

        reward = -dist  # main signal

        if dist < 0.2:
            reward += 20.0

        return reward

    # =========================
    def _check_done(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        xy = np.array(pos[:2])

        return np.linalg.norm(xy - self.goal) < 0.2