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

        # Action = [forward, turn]
        # continuous control → smoother than discrete
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # 🔥 CHANGED: observation now includes angle-to-goal info
        # before: agent had to "figure out" angle from raw data
        # now: we GIVE it directly → much easier learning
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(7,),
            dtype=np.float32
        )

        self.robot = None
        self.goal = None
        self.steps = 0
        self.prev_dist = None

        # 🔥 NEW: used for action smoothing (reduces jitter)
        self.prev_action = np.zeros(2)

    # =========================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("r2d2.urdf", [0, 0, 0.1])

        self.steps = 0
        self.prev_dist = None
        self.prev_action = np.zeros(2)

        # random goal → generalization
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

        # 🔥 NEW: action smoothing
        # prevents jerky motion → more stable learning
        action = 0.7 * self.prev_action + 0.3 * action
        self.prev_action = action

        forward, turn = action

        forward_speed = forward * 2.5
        turn_speed = turn * 1.2

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        rot = p.getMatrixFromQuaternion(orn)

        # robot forward direction in world frame
        forward_dir = np.array([rot[0], rot[3], rot[6]], dtype=np.float32)
        velocity = forward_dir * forward_speed

        # apply velocity
        p.resetBaseVelocity(
            self.robot,
            linearVelocity=velocity.tolist(),
            angularVelocity=[0, 0, turn_speed]
        )

        p.stepSimulation()

        if self.render:
            time.sleep(1 / 240)

        obs = self._get_obs()

        # reward now uses improved shaping
        reward = self._compute_reward(action)

        done = self._check_done()
        truncated = self.steps >= 400  # 🔥 longer episodes = better learning

        return obs, reward, done, truncated, {}

    # =========================
    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        xy = np.array(pos[:2], dtype=np.float32)

        yaw = p.getEulerFromQuaternion(orn)[2]

        goal_vec = self.goal - xy
        dist = np.linalg.norm(goal_vec)

        # normalized direction to goal
        goal_dir = goal_vec / (dist + 1e-6)

        # 🔥 CRITICAL CHANGE:
        # compute relative angle between robot heading and goal
        goal_angle = np.arctan2(goal_vec[1], goal_vec[0])
        angle_diff = goal_angle - yaw

        # normalize angle → avoids discontinuity at ±pi
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        linear_vel, angular_vel = p.getBaseVelocity(self.robot)

        return np.array([
            goal_dir[0],                  # where goal is (x)
            goal_dir[1],                  # where goal is (y)

            # 🔥 HUGE IMPROVEMENT:
            # instead of raw angle → use cos/sin
            # makes learning smooth + avoids wrap issues
            np.cos(angle_diff),
            np.sin(angle_diff),

            linear_vel[0] / 3.0,
            linear_vel[1] / 3.0,
            angular_vel[2] / 3.0
        ], dtype=np.float32)

    # =========================
    def _compute_reward(self, action):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        xy = np.array(pos[:2], dtype=np.float32)

        dist = np.linalg.norm(xy - self.goal)

        if self.prev_dist is None:
            self.prev_dist = dist

        # 🔥 CORE REWARD SIGNAL:
        # reward progress toward goal
        progress = self.prev_dist - dist
        self.prev_dist = dist

        reward = 15.0 * progress  
        # ↑ stronger than before → clearer learning signal

        # -------------------------
        # 🔥 MOST IMPORTANT CHANGE
        # -------------------------
        # reward pointing toward the goal

        yaw = p.getEulerFromQuaternion(orn)[2]
        goal_angle = np.arctan2(self.goal[1] - xy[1], self.goal[0] - xy[0])
        angle_diff = goal_angle - yaw
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        # 🔥 THIS is what makes the agent "face the right direction"
        reward += 1.0 * np.cos(angle_diff)

        # WHY THIS WORKS:
        # cos(angle_diff) =
        #   +1 → perfectly facing goal
        #   0  → sideways
        #  -1 → facing away
        #
        # → agent learns:
        # "I should rotate until this becomes +1"

        # -------------------------
        # discourage useless spinning
        reward -= 0.05 * abs(action[1])

        # small time penalty
        reward -= 0.01

        # success reward
        if dist < 0.2:
            reward += 150.0

        return float(reward)

    # =========================
    def _check_done(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        xy = np.array(pos[:2], dtype=np.float32)

        return np.linalg.norm(xy - self.goal) < 0.2