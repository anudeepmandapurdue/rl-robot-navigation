import pybullet as p 
import pybullet_data
import time

def run_sim():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.loadURDF("plane.urdf")
    robot = p.loadURDF("r2d2.urdf", [0,0,0.1])

    for _ in range(10000):
        p.resetBaseVelocity(robot, linearVelocity = [1,1,0.5])
        p.stepSimulation()
        time.sleep(1/240)

    p.disconnect


if __name__ == "__main__":
    run_sim()