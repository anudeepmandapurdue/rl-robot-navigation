import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time
import os


class NavigationEnv(gym.Env):
    def __init__(self, render=True):
        super().__init__()

        self.render = render
        self.client = p.connect(p.GUI if render else p.DIRECT)

        self.data_path = pybullet_data.getDataPath()
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        self.num_obstacles = 5
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(7 + self.num_obstacles * 3,),
            dtype=np.float32
        )

        self.robot = None
        self.goal = None
        self.obstacles = []
        self.left_wheel = None
        self.right_wheel = None

        self.steps = 0
        self.prev_dist = None
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.max_steps = 400

    # =========================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        p.loadURDF(os.path.join(self.data_path, "plane.urdf"))

        urdf_path = os.path.join(
            self.BASE_DIR,
            "assets",
            "turtlebot3",
            "turtlebot3_description",
            "urdf",
            "turtlebot3_burger.urdf"
        )

        mesh_path = os.path.join(
            self.BASE_DIR,
            "assets",
            "turtlebot3",
            "turtlebot3_description"
        )

        p.setAdditionalSearchPath(mesh_path)

        self.robot = p.loadURDF(urdf_path, [0, 0, 0.1])

        p.changeDynamics(
            self.robot,
            -1,
            linearDamping=0.1,
            angularDamping=0.1,
            lateralFriction=0.8,
            restitution=0.0
        )

        for i in range(p.getNumJoints(self.robot)):
            p.changeDynamics(
                self.robot,
                i,
                linearDamping=0,
                angularDamping=0,
                lateralFriction=0.9
            )

        self.left_wheel = None
        self.right_wheel = None

        for i in range(p.getNumJoints(self.robot)):
            name = p.getJointInfo(self.robot, i)[1].decode("utf-8")
            if "left" in name:
                self.left_wheel = i
            if "right" in name:
                self.right_wheel = i

        self.steps = 0
        self.prev_dist = None
        self.prev_action = np.zeros(2, dtype=np.float32)

        self.np_random = np.random.default_rng(seed)
        self.goal = self.np_random.uniform(-1.0, 1.0, size=2).astype(np.float32)

        p.loadURDF(
            os.path.join(self.data_path, "sphere_small.urdf"),
            [self.goal[0], self.goal[1], 0.1],
            globalScaling=2.0
        )

        self.obstacles = []

        for _ in range(self.num_obstacles):
            while True:
                obs_pos = self.np_random.uniform(-0.7, 0.7, size=2).astype(np.float32)
                too_close_to_robot = np.linalg.norm(obs_pos) < 0.4
                too_close_to_goal = np.linalg.norm(obs_pos - self.goal) < 0.4
                too_close_to_others = any(
                    np.linalg.norm(obs_pos - np.array(p.getBasePositionAndOrientation(o)[0][:2])) < 0.4
                    for o in self.obstacles
                )
                if not too_close_to_robot and not too_close_to_goal and not too_close_to_others:
                    break

            obs_id = p.loadURDF(
                os.path.join(self.data_path, "cube_small.urdf"),
                [obs_pos[0], obs_pos[1], 0.1],
                globalScaling=1.5,
                useFixedBase=True  # CRITICAL: makes obstacle immovable
            )

            # Make obstacle high friction and no bounce so robot can't push through
            p.changeDynamics(
                obs_id,
                -1,
                lateralFriction=1.0,
                restitution=0.0,
                mass=0  # mass=0 = static, cannot be moved by any force
            )

            self.obstacles.append(obs_id)

        return self._get_obs(), {}

    # =========================
    def step(self, action):
        self.steps += 1

        action = np.clip(action, -1, 1)

        action = 0.4 * self.prev_action + 0.6 * action
        self.prev_action = action

        forward, turn = action
        speed = 15.0  # reduced from 25 — high speed overpowers collision physics

        left = (forward - 0.4 * turn) * speed
        right = (forward + 0.4 * turn) * speed

        p.setJointMotorControl2(
            self.robot,
            self.left_wheel,
            p.VELOCITY_CONTROL,
            targetVelocity=left,
            force=800  # reduced from 5000 — lower force respects physics collisions
        )
        p.setJointMotorControl2(
            self.robot,
            self.right_wheel,
            p.VELOCITY_CONTROL,
            targetVelocity=right,
            force=800
        )

        for _ in range(4):
            p.stepSimulation()

        if self.render:
            time.sleep(1 / 240)

        obs = self._get_obs()
        reward = self._compute_reward(action)
        done = self._check_done()
        truncated = self.steps >= self.max_steps

        return obs, reward, done, truncated, {}

    # =========================
    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        xy = np.array(pos[:2], dtype=np.float32)

        yaw = p.getEulerFromQuaternion(orn)[2]

        goal_vec = self.goal - xy
        dist = np.linalg.norm(goal_vec)
        goal_dir = goal_vec / (dist + 1e-6)

        goal_angle = np.arctan2(goal_vec[1], goal_vec[0])
        angle_diff = goal_angle - yaw
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        linear_vel, angular_vel = p.getBaseVelocity(self.robot)
        lin_speed = np.linalg.norm(linear_vel[:2])
        lin_dir = np.array(linear_vel[:2]) / (lin_speed + 1e-6)

        base_obs = np.array([
            goal_dir[0],
            goal_dir[1],
            np.cos(angle_diff),
            np.sin(angle_diff),
            np.clip(lin_speed / 5.0, -1, 1),
            np.clip(lin_dir[0], -1, 1),
            np.clip(angular_vel[2] / 5.0, -1, 1)
        ], dtype=np.float32)

        obs_vecs = []
        for obs_id in self.obstacles:
            obs_pos_world, _ = p.getBasePositionAndOrientation(obs_id)
            rel = np.array(obs_pos_world[:2], dtype=np.float32) - xy
            rel_norm = np.clip(rel / 2.0, -1, 1)
            obs_dist = np.linalg.norm(rel)
            closeness = np.clip(1.0 - obs_dist / 0.8, 0.0, 1.0)
            obs_vecs.extend([rel_norm[0], rel_norm[1], closeness])

        return np.concatenate([base_obs, np.array(obs_vecs, dtype=np.float32)])

    # =========================
    def _check_collision(self):
        # Use real PyBullet contact detection instead of distance approximation
        for obs_id in self.obstacles:
            contacts = p.getContactPoints(self.robot, obs_id)
            if contacts and len(contacts) > 0:
                return True
        return False

    # =========================
    def _compute_reward(self, action):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        xy = np.array(pos[:2], dtype=np.float32)

        dist = np.linalg.norm(xy - self.goal)

        if self.prev_dist is None:
            self.prev_dist = dist

        progress = self.prev_dist - dist
        self.prev_dist = dist

        reward = 20.0 * progress

        yaw = p.getEulerFromQuaternion(orn)[2]
        goal_angle = np.arctan2(self.goal[1] - xy[1], self.goal[0] - xy[0])
        angle_diff = np.arctan2(np.sin(goal_angle - yaw), np.cos(goal_angle - yaw))
        reward += 0.8 * np.cos(angle_diff)

        _, angular_vel = p.getBaseVelocity(self.robot)
        reward -= 0.03 * abs(angular_vel[2])

        reward -= 0.001
        reward -= 0.01 * dist

        # Success
        if dist < 0.2:
            speed_bonus = 40.0 * (1.0 - self.steps / self.max_steps)
            return float(reward + 300.0 + speed_bonus)

        # Real contact penalty
        if self._check_collision():
            reward -= 100.0

        # Soft proximity penalty
        for obs_id in self.obstacles:
            obs_pos_world, _ = p.getBasePositionAndOrientation(obs_id)
            obs_xy = np.array(obs_pos_world[:2], dtype=np.float32)
            obs_dist = np.linalg.norm(xy - obs_xy)
            if obs_dist < 0.5:
                reward -= 6.0 * (0.4 - obs_dist)

        return float(reward)

    # =========================
    def _check_done(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        xy = np.array(pos[:2], dtype=np.float32)

        return np.linalg.norm(xy - self.goal) < 0.2