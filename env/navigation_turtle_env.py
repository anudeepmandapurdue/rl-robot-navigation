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

        self.num_obstacles = 2
        self.obstacle_speed = 0.15

        # 7 base + per obstacle: rel_x, rel_y, closeness, vel_x, vel_y
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(7 + self.num_obstacles * 5,),
            dtype=np.float32
        )

        self.robot = None
        self.goal = None
        self.obstacles = []
        self.obstacle_velocities = []
        self.left_wheel = None
        self.right_wheel = None

        self.steps = 0
        self.prev_dist = None
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.max_steps = 300

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
        self.obstacle_velocities = []

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
                useFixedBase=False
            )

            p.changeDynamics(
                obs_id,
                -1,
                lateralFriction=0.0,
                restitution=0.0,
                linearDamping=0.0,
                angularDamping=0.0
            )

            angle = self.np_random.uniform(0, 2 * np.pi)
            vel = np.array([np.cos(angle), np.sin(angle)]) * self.obstacle_speed
            self.obstacle_velocities.append(vel)
            self.obstacles.append(obs_id)

        return self._get_obs(), {}

    # =========================
    def _update_obstacles(self):
        bounds = 0.9

        for i, obs_id in enumerate(self.obstacles):
            pos, _ = p.getBasePositionAndOrientation(obs_id)
            xy = np.array(pos[:2])
            vel = self.obstacle_velocities[i].copy()

            # Bounce off boundaries
            if abs(xy[0]) > bounds:
                vel[0] *= -1
            if abs(xy[1]) > bounds:
                vel[1] *= -1

            self.obstacle_velocities[i] = vel

            p.resetBaseVelocity(
                obs_id,
                linearVelocity=[vel[0], vel[1], 0],
                angularVelocity=[0, 0, 0]
            )

    # =========================
    def step(self, action):
        self.steps += 1

        action = np.clip(action, -1, 1)
        action = 0.3 * self.prev_action + 0.7 * action
        self.prev_action = action

        forward, turn = action
        speed = 12.0

        # increased turn scaling from 0.4 to 0.6
        # so robot can rotate fast enough to align before moving
        left = (forward - 0.6 * turn) * speed
        right = (forward + 0.6 * turn) * speed

        p.setJointMotorControl2(
            self.robot, self.left_wheel,
            p.VELOCITY_CONTROL,
            targetVelocity=left,
            force=800
        )
        p.setJointMotorControl2(
            self.robot, self.right_wheel,
            p.VELOCITY_CONTROL,
            targetVelocity=right,
            force=800
        )

        self._update_obstacles()

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
        for i, obs_id in enumerate(self.obstacles):
            obs_pos_world, _ = p.getBasePositionAndOrientation(obs_id)
            rel = np.array(obs_pos_world[:2], dtype=np.float32) - xy
            rel_norm = np.clip(rel / 2.0, -1, 1)
            obs_dist = np.linalg.norm(rel)
            closeness = np.clip(1.0 - obs_dist / 0.8, 0.0, 1.0)

            vel = self.obstacle_velocities[i]
            vel_norm = np.clip(vel / self.obstacle_speed, -1, 1)

            obs_vecs.extend([rel_norm[0], rel_norm[1], closeness, vel_norm[0], vel_norm[1]])

        return np.concatenate([base_obs, np.array(obs_vecs, dtype=np.float32)])

    # =========================
    def _check_collision(self):
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

        yaw = p.getEulerFromQuaternion(orn)[2]
        goal_angle = np.arctan2(self.goal[1] - xy[1], self.goal[0] - xy[0])
        angle_diff = np.arctan2(np.sin(goal_angle - yaw), np.cos(goal_angle - yaw))

        _, angular_vel = p.getBaseVelocity(self.robot)
        aligned = abs(angle_diff) < 0.3  # robot is roughly facing the goal

        # ---- PHASE 1: ALIGNMENT ----
        # when not aligned, heavily reward turning toward goal
        # and penalize moving forward so robot rotates first
        if not aligned:
            # STRONG heading reward — rotating toward goal is the only goal right now
            reward = 2.0 * np.cos(angle_diff)

            # PENALIZE forward motion when not aligned — rotate first, move second
            reward -= 0.5 * abs(action[0])

            # PENALIZE spinning in wrong direction
            reward -= 0.05 * abs(angular_vel[2])

            # small time penalty
            reward -= 0.005

        # ---- PHASE 2: MOVE TOWARD GOAL ----
        # once aligned, switch to progress-based reward
        else:
            # PROGRESS — strong forward signal once aligned
            # higher = faster approach, lower = more cautious
            reward = 10.0 * progress

            # HEADING maintenance — small bonus to stay aligned while moving
            # keep low so robot doesn't stop to re-align constantly
            reward += 0.3 * np.cos(angle_diff)

            # PENALIZE spinning while moving — robot should drive straight
            reward -= 0.15 * abs(angular_vel[2])

            # DISTANCE PENALTY — urgency to close gap
            reward -= 0.05 * dist

            # TIME PENALTY
            reward -= 0.005

        # ---- PHASE 3: OBSTACLE AVOIDANCE ----
        # always active regardless of phase
        # check nearest obstacle and react

        nearest_obs_dist = float('inf')
        for obs_id in self.obstacles:
            obs_pos_world, _ = p.getBasePositionAndOrientation(obs_id)
            obs_xy = np.array(obs_pos_world[:2], dtype=np.float32)
            obs_dist = np.linalg.norm(xy - obs_xy)
            if obs_dist < nearest_obs_dist:
                nearest_obs_dist = obs_dist

        # when obstacle is close, temporarily override heading penalty
        # so robot can turn freely to avoid it
        obstacle_nearby = nearest_obs_dist < 0.5

        if obstacle_nearby:
            # RELEASE turn penalty — robot needs to steer freely to avoid
            reward += 0.05 * abs(action[1])  # small bonus for actually steering

            # PROXIMITY PENALTY — scale with closeness
            # 0.35 = danger zone, 0.5 = penalty strength
            for obs_id in self.obstacles:
                obs_pos_world, _ = p.getBasePositionAndOrientation(obs_id)
                obs_xy = np.array(obs_pos_world[:2], dtype=np.float32)
                obs_dist = np.linalg.norm(xy - obs_xy)
                if obs_dist < 0.35:
                    reward -= 0.5 * (0.35 - obs_dist)

        # COLLISION PENALTY — always active
        # increase if robot still hits obstacles
        # decrease if robot is too afraid to move
        if self._check_collision():
            reward -= 5.0

        # SUCCESS BONUS — reaching goal
        # speed_bonus encourages finishing quickly
        if dist < 0.2:
            speed_bonus = 5.0 * (1.0 - self.steps / self.max_steps)
            return float(reward + 30.0 + speed_bonus)

        return float(reward)
    # =========================
    def _check_done(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        xy = np.array(pos[:2], dtype=np.float32)

        return np.linalg.norm(xy - self.goal) < 0.1