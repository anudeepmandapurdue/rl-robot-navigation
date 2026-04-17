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

        # Observation: fully normalized [-1, 1] except yaw in [-1, 1]
        self.observation_space = gym.spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.robot = None
        self.goal = None
        self.steps = 0
        self.prev_dist = None

    # =========================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("r2d2.urdf", [0, 0, 0.1])

        self.steps = 0
        self.prev_dist = None

        self.np_random = np.random.default_rng(seed)
        self.goal = self.np_random.uniform(-2, 2, size=2).astype(np.float32)

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

        forward_speed = forward * 2.0
        turn_speed = turn * 1.0

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        rot = p.getMatrixFromQuaternion(orn)

        forward_dir = np.array([rot[0], rot[3], rot[6]], dtype=np.float32)
        velocity = forward_dir * forward_speed

        p.resetBaseVelocity(
            self.robot,
            linearVelocity=velocity.tolist(),
            angularVelocity=[0.0, 0.0, turn_speed]
        )

        p.stepSimulation()

        if self.render:
            time.sleep(1 / 240)

        obs = self._get_obs()
        reward = self._compute_reward()
        done = self._check_done()

        truncated = self.steps >= 200

        return obs, reward, done, truncated, {}

    # =========================
    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        xy = np.array(pos[:2], dtype=np.float32)

        yaw = p.getEulerFromQuaternion(orn)[2] / np.pi  # normalize

        goal_vec = self.goal - xy

        # normalize by environment scale (goal in [-2,2])
        direction = np.clip(goal_vec / 2.0, -1, 1)

        linear_vel, angular_vel = p.getBaseVelocity(self.robot)

        linear_vel = np.array(linear_vel[:2], dtype=np.float32) / 3.0
        angular_z = np.float32(angular_vel[2]) / 3.0

        return np.array([
            direction[0],
            direction[1],
            yaw,
            linear_vel[0],
            linear_vel[1],
            angular_z
        ], dtype=np.float32)

    # =========================
    def _compute_reward(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        xy = np.array(pos[:2], dtype=np.float32)

        dist = np.linalg.norm(xy - self.goal)

        # initialize prev distance
        if self.prev_dist is None:
            self.prev_dist = dist

        # FIX: correct progress calculation
        progress = self.prev_dist - dist
        self.prev_dist = dist

        progress = np.clip(progress, -0.2, 0.2)

        reward = progress * 20.0

        # small shaping penalty (distance pressure)
        reward -= 0.005 * dist

        # success bonus handled in done, not reward logic
        if dist < 0.2:
            reward += 100.0

        return float(reward)

    # =========================
    def _check_done(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        xy = np.array(pos[:2], dtype=np.float32)

        return np.linalg.norm(xy - self.goal) < 0.2