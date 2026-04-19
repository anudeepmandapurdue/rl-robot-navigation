import pybullet as p
import pybullet_data
import time
import os

# Connect
p.connect(p.GUI)

# ----------------------------
# 1. Built-in PyBullet assets
# ----------------------------
data_path = pybullet_data.getDataPath()
p.setAdditionalSearchPath(data_path)

# Load ground
plane = p.loadURDF(os.path.join(data_path, "plane.urdf"))

# ----------------------------
# 2. TurtleBot3 assets (LOCAL)
# ----------------------------

# IMPORTANT: change this if your folder is elsewhere
turtlebot_path = "assets/turtlebot3/turtlebot3_description/urdf"

p.setAdditionalSearchPath(turtlebot_path)

# Load robot (explicit file reference)
robot = p.loadURDF("turtlebot3_burger.urdf", [0, 0, 0.1])

# ----------------------------
# 3. Physics
# ----------------------------
p.setGravity(0, 0, -9.81)

# Print joints so we can control it later
print("\n--- JOINT INFO ---")
for i in range(p.getNumJoints(robot)):
    print(p.getJointInfo(robot, i))

# ----------------------------
# 4. Simulation loop
# ----------------------------
while True:
    p.stepSimulation()
    time.sleep(1./240.)