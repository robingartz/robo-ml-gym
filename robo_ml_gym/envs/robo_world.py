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

    def __init__(self, render_mode=None):
        # box dimensions (the area the cube can spawn within)
        BOX_WIDTH = 0.39 / 2
        BOX_LENGTH = 0.58 / 2
        BOX_HEIGHT = 0.18 / 2  # height of the sides of the box from its base
        BOX_OFFSET = 0.008 / 2  # thickness of the sides of the box
        FLAT = 0.1

        # defines a centre point (x,y,z) and the (w/2,l/2,h/2) for a cube region in space for the possible starting cube
        # position and target position
        self.TARGET_REGION = np.array([[BOX_WIDTH+0.52, 0.0, 0.273], [BOX_WIDTH*2, BOX_LENGTH*1.2, 0.0+FLAT]], dtype=np.float32)
        self.CUBE_REGION = np.array([[BOX_WIDTH+0.52, 0.0, 0.273], [BOX_WIDTH+0.52, 0.0+FLAT, 0.273]], dtype=np.float32)

        # get the low and high limits for these regions
        # target region
        TR = self.TARGET_REGION
        self.TARGET_REGION_LOW = np.array([TR[0][0] - TR[1][0], TR[0][1] - TR[1][1], TR[0][2] - TR[1][2]])
        self.TARGET_REGION_HIGH = np.array([TR[0][0] + TR[1][0], TR[0][1] + TR[1][1], TR[0][2] + TR[1][2]])
        # cube spawn region
        CR = self.CUBE_REGION
        self.CUBE_REGION_LOW = np.array([CR[0][0] - CR[1][0], CR[0][1] - CR[1][1], CR[0][2] - CR[1][2]])
        self.CUBE_REGION_HIGH = np.array([CR[0][0] + CR[1][0], CR[0][1] + CR[1][1], CR[0][2] + CR[1][2]])
        # end effector possible position
        MAX_EF_HEIGHT = 0.4  # this value is debatable, should it be enforced etc?
        self.END_EFFECTOR_REGION_HIGH = np.array([TR[0][0] + TR[1][0], TR[0][1] + TR[1][1], TR[0][2] + MAX_EF_HEIGHT])

        # observations are dictionaries with the robot's joint angles and end effector position and
        # the cube's location and target location
        # each joint is between the min and max positions for the joint
        # the positions are a region in space encoded as the min(x,y,z) and max(x,y,z) for the box's region
        self.observation_space = spaces.Dict(
            {
                "joints": spaces.Box(0, 360, shape=(6,), dtype=np.float32),  # ToDo: update limits according to getJointInfo() values
                "end_effector_pos": spaces.Box(self.TARGET_REGION_LOW, self.END_EFFECTOR_REGION_HIGH, shape=(3,), dtype=np.float32),
                "cube_pos": spaces.Box(self.CUBE_REGION_LOW, self.CUBE_REGION_HIGH, shape=(3,), dtype=np.float32),
                "target_pos": spaces.Box(self.TARGET_REGION_LOW, self.TARGET_REGION_HIGH, shape=(3,), dtype=np.float32)
            }
        )

        # we have 6 actions, corresponding to the angles of the six joints
        self.action_space = spaces.Box(0, 2*np.pi, shape=(6,), dtype=np.float32)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction each of the joints will move in if that action is taken.
        I.e. 0 corresponds to joint 0: -0.5 deg, joint 0: +0.5 deg, joint 1: -0.5 deg, etc.
        """
        #self._action_to_angle_cmd = {
        #    0: np.array([1, 0]),
        #    1: np.array([0, 1]),
        #}

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode  # human or rgb_array

        self.physics_client = None
        self.cube_id = None
        self.robot_id = None
        self.joints_count = None
        self._end_effector_pos = None
        self._target_location = None

    def _get_obs(self):
        #joints_info = []
        #_jointIndexIdx = 0
        #_jointNameIdx = 1
        #_parentFramePosIdx = 14
        #joints_info = [pybullet.getJointInfo(self.robot_id, joint_num)[_parentFramePosIdx] for joint_num in range(joints_count)]

        # observations
        joint_positions = np.array([info[0] for info in pybullet.getJointStates(self.robot_id, range(self.joints_count))], dtype=np.float32)
        # ToDo: below is wrong, this is the joint position, but it should be the end effector position
        self._end_effector_pos = np.array(pybullet.getJointState(self.robot_id, self.joints_count-1)[0], dtype=np.float32)
        cube_pos, cube_orn = pybullet.getBasePositionAndOrientation(self.cube_id)

        observations = {
            "joints": joint_positions,
            "end_effector_pos": self._end_effector_pos,
            "cube_pos": np.array(cube_pos, dtype=np.float32),
            "target_pos": np.array(self._target_location, dtype=np.float32)
        }

        #print(observations)
        return observations

    def _get_info(self):
        return {"distance": np.linalg.norm(self._target_location - self._end_effector_pos, ord=1)}

    def reset(self, seed=None, options=None):
        # we need the following line to seed self.np_random
        super().reset(seed=seed)

        # set up the environment if not yet done
        if self.physics_client is None:
            self._setup()

        # reset the robot's position
        pybullet.restoreState(self.init_state)

        # set a random position for the cube
        cube_pos = np.array([self.np_random.uniform(self.CUBE_REGION_LOW[0], self.CUBE_REGION_HIGH[0]),
                             self.np_random.uniform(self.CUBE_REGION_LOW[1], self.CUBE_REGION_HIGH[1]),
                             self.np_random.uniform(self.CUBE_REGION_LOW[2], self.CUBE_REGION_HIGH[2])
                             ])
        cub_orn = (0, 0, 0, 0)
        pybullet.resetBasePositionAndOrientation(self.cube_id, cube_pos, cub_orn)

        # set a random position for the target location
        self._target_location = np.array([self.np_random.uniform(self.TARGET_REGION_LOW[0], self.TARGET_REGION_HIGH[0]),
                                          self.np_random.uniform(self.TARGET_REGION_LOW[1], self.TARGET_REGION_HIGH[1]),
                                          self.np_random.uniform(self.TARGET_REGION_LOW[2], self.TARGET_REGION_HIGH[2])
                                          ])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3,4,5}) to the direction we walk in
        #direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._end_effector_pos, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        # move joint
        pybullet.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=0, controlMode=pybullet.VELOCITY_CONTROL,
                                       targetVelocity=0.5)

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.physics_client is None:
            self._setup()

        if self.render_mode == "human":
            ...

        pybullet.stepSimulation()

        if self.render_mode == "human":
            time.sleep(1.0/240.0)

    def _setup(self):
        # pybullet setup
        self.physics_client = pybullet.connect(pybullet.GUI)  # pybullet.GUI or pybullet.DIRECT
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

        # world building
        pybullet.setGravity(0, 0, -9.81)

        # ground
        plane_id = pybullet.loadURDF("plane.urdf")

        # visual only objects
        if self.render_mode == "human":
            # baseplate
            plate_id = pybullet.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=[0.01, 0.1, 0.45])
            mass = 0
            pybullet.createMultiBody(mass, plate_id, basePosition=[-0.01, 0.0, 0.45])

            # ABB circuit box
            abb_box_id = pybullet.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=[0.1, 0.2, 0.1])
            mass = 0
            pybullet.createMultiBody(mass, abb_box_id, basePosition=[0.1, 0.1, 0.1])

        # box (an urdf file will not have accurate hit-boxes as it will fill in the empty space)
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
        self.cube_id = pybullet.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=[0.05/2, 0.05/2, 0.05/2])
        mass = 0.03
        self.cube_id = pybullet.createMultiBody(mass, self.cube_id, basePosition=[0.70, 0.0, 0.70])
        cube_pos, cube_orn = pybullet.getBasePositionAndOrientation(self.cube_id)
        print("cube_pos, cube_orn:", cube_pos, cube_orn)

        # ABB IRB120
        start_pos = [0, 0, 0.18/2 + 0.65]
        start_orientation = pybullet.getQuaternionFromEuler([0, np.pi/2, 0])
        urdf_path = "robo_ml_gym/models/irb120/irb120.urdf"
        if "robo_ml_gym" not in os.listdir():  # if cwd is 1 level up, then prepend gym-examples/ dir
            urdf_path = "/robo_ml_gym/" + urdf_path
        self.robot_id = pybullet.loadURDF(urdf_path, basePosition=start_pos, baseOrientation=start_orientation,
                                          useFixedBase=1, flags=pybullet.URDF_MAINTAIN_LINK_ORDER)
        self.joints_count = pybullet.getNumJoints(self.robot_id)
        print(pybullet.getNumJoints(self.robot_id))
        print(pybullet.getJointInfo(self.robot_id, 0))
        # set the center of mass frame (loadURDF sets base link frame) start_pos/Ornp.resetBasePositionAndOrientation(robot_id, start_pos, start_orientation)

        self.init_state = pybullet.saveState()

        #pybullet.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=1,
        #                               controlMode=pybullet.POSITION_CONTROL, targetPosition=-0.9)

    def close(self):
        if self.physics_client is not None:
            robot_pos, robot_orn = pybullet.getBasePositionAndOrientation(self.robot_id)
            print(robot_pos, robot_orn)
            pybullet.disconnect()
