# standard libs
import time
import os

# 3rd party libs
import numpy as np
import pybullet
import pybullet_data

# gym
import gymnasium as gym
from gymnasium import spaces


class RoboWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 14}

    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 18 actions, corresponding to CW/CCW/ six joints
        self.action_space = spaces.Discrete(4)
        #self.action_space = spaces.Continuous(6)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = "human"  # human or rgb_array

        self.physics_client = None

    def _get_obs(self):
        print({"agent": self._agent_location, "target": self._target_location})
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3,4,5}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # move joint
        #pybullet.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=0, controlMode=pybullet.VELOCITY_CONTROL,
        #                               targetVelocity=0.5)

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.physics_client is None and self.render_mode == "human":
            # pybullet setup
            self.physics_client = pybullet.connect(pybullet.GUI)  # pybullet.GUI or pybullet.DIRECT
            pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

            # world building
            pybullet.setGravity(0, 0, -9.81)

            # ground
            plane_id = pybullet.loadURDF("plane.urdf")
            
            # baseplate
            plate_id = pybullet.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=[0.01, 0.1, 0.45])
            mass = 0
            pybullet.createMultiBody(mass, plate_id, basePosition=[-0.01, 0.0, 0.45])

            # ABB circuit box
            abb_box_id = pybullet.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=[0.1, 0.2, 0.1])
            mass = 0
            pybullet.createMultiBody(mass, abb_box_id, basePosition=[0.1, 0.1, 0.1])

            # box
            mass = 0
            BOX_WIDTH = 0.39 / 2
            BOX_LENGTH = 0.58 / 2
            BOX_HEIGHT = 0.18 / 2
            BOX_OFFSET = 0.008 / 2
            BOX_POS = (BOX_WIDTH+0.52, 0.0, 0.273)
            shape_types = [pybullet.GEOM_BOX] * 5
            half_extents = [[BOX_WIDTH-BOX_OFFSET, BOX_LENGTH, BOX_OFFSET],
                            [BOX_WIDTH-BOX_OFFSET, BOX_OFFSET, BOX_HEIGHT-BOX_OFFSET],
                            [BOX_WIDTH-BOX_OFFSET, BOX_OFFSET, BOX_HEIGHT-BOX_OFFSET],
                            [BOX_OFFSET, BOX_LENGTH, BOX_HEIGHT],
                            [BOX_OFFSET, BOX_LENGTH, BOX_HEIGHT]]
            collision_frame_positions = [[BOX_POS[0], BOX_POS[1], BOX_POS[2]+BOX_OFFSET],
                                         [BOX_POS[0], BOX_POS[1]-BOX_LENGTH+BOX_OFFSET, BOX_POS[2]+BOX_HEIGHT+BOX_OFFSET],
                                         [BOX_POS[0], BOX_POS[1]+BOX_LENGTH-BOX_OFFSET, BOX_POS[2]+BOX_HEIGHT+BOX_OFFSET],
                                         [BOX_POS[0]-BOX_WIDTH, BOX_POS[1], BOX_POS[2]+BOX_HEIGHT],
                                         [BOX_POS[0]+BOX_WIDTH, BOX_POS[1], BOX_POS[2]+BOX_HEIGHT]]
            box_id = pybullet.createCollisionShapeArray(shapeTypes=shape_types, halfExtents=half_extents, collisionFramePositions=collision_frame_positions)
            pybullet.createMultiBody(mass, box_id, basePosition=[0,0,0])

            # objects for pick-n-place
            ob1_id = pybullet.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=[0.05/2, 0.05/2, 0.05/2])
            mass = 0.03
            pybullet.createMultiBody(mass, ob1_id, basePosition=[0.70, 0.0, 0.70])

            # ABB IRB120
            start_pos = [0, 0, 0.18/2 + 0.65]
            start_orientation = pybullet.getQuaternionFromEuler([0, np.pi/2, 0])
            urdf_path = "robo_ml_gym/models/irb120/irb120.urdf"
            if "robo_ml_gym" not in os.listdir():  # if cwd is 1 level up, then prepend gym-examples/ dir
                urdf_path = "/robo_ml_gym/" + urdf_path
            self.robot_id = pybullet.loadURDF(urdf_path, basePosition=start_pos, baseOrientation=start_orientation,
                                              useFixedBase=1, flags=pybullet.URDF_MAINTAIN_LINK_ORDER)
            print(pybullet.getNumJoints(self.robot_id))
            print(pybullet.getJointInfo(self.robot_id, 0))
            # set the center of mass frame (loadURDF sets base link frame) start_pos/Ornp.resetBasePositionAndOrientation(robot_id, start_pos, start_orientation)

            pybullet.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=1,
                                           controlMode=pybullet.POSITION_CONTROL, targetPosition=-0.9)

        pybullet.stepSimulation()

        if self.render_mode == "human":
            time.sleep(1.0/240.0)

    def close(self):
        if self.physics_client is not None:
            robot_pos, robot_orn = pybullet.getBasePositionAndOrientation(self.robot_id)
            print(robot_pos, robot_orn)
            pybullet.disconnect()
