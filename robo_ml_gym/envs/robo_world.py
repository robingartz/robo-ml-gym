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

# box dimensions (the area the cube can spawn within)
BOX_WIDTH = 0.39 / 2
BOX_LENGTH = 0.58 / 2
BOX_HEIGHT = 0.18 / 2  # height of the sides of the box from its base
BOX_OFFSET = 0.008 / 2  # thickness of the sides of the box
BOX_POS = (BOX_WIDTH+0.52, 0.0, 0.273-0.273)  # box on ground
#BOX_POS = (BOX_WIDTH + 0.42, 0.0, 0.273)  # box above ground and closer to robot
#BOX_POS = (BOX_WIDTH+0.52, 0.0, 0.12)  # box above ground and closer to robot
CUBE_DIM = 0.05


class RoboWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 14}

    def __init__(self, render_mode=None, verbose=False, total_steps=None, fname_app=""):
        # box dimensions (the area the cube can spawn within)
        FLAT = 0.1  # ToDo: figure out what to do for the regions that are 2D...
        MAX_EF_HEIGHT = 0.6  # ToDo: this value is debatable, should it be enforced etc?
        MAX_JOINT_VEL = 10.0  # max joint velocity

        # defines a centre point (x,y,z) and the (w/2,l/2,h/2) for a cube region in space for the possible starting cube
        # position and target position
        self.TARGET_REGION = np.array([[BOX_POS[0], BOX_POS[1], BOX_POS[2]+BOX_HEIGHT],
                                       [BOX_WIDTH, BOX_LENGTH*1.6, 0.0+FLAT]], dtype=np.float32)
        self.CUBE_START_REGION = np.array([[BOX_POS[0], BOX_POS[1], BOX_POS[2]+BOX_HEIGHT],
                                           [BOX_WIDTH-CUBE_DIM, BOX_LENGTH-CUBE_DIM, BOX_HEIGHT-CUBE_DIM/2]],  # ToDo: CUBE_DIM / 2
                                          dtype=np.float32)
        self.CUBE_REGION = np.array([[BOX_POS[0], BOX_POS[1], BOX_POS[2]+BOX_HEIGHT],
                                     [BOX_WIDTH, BOX_LENGTH*1.6, BOX_HEIGHT*3]], dtype=np.float32)

        # get the low and high limits for these regions
        # target region
        TR = self.TARGET_REGION
        self.TARGET_REGION_LOW = np.array([TR[0][0] - TR[1][0], TR[0][1] - TR[1][1], TR[0][2] - TR[1][2]])
        self.TARGET_REGION_HIGH = np.array([TR[0][0] + TR[1][0], TR[0][1] + TR[1][1], TR[0][2] + TR[1][2]])
        # cube spawn region
        CSR = self.CUBE_START_REGION
        self.CUBE_START_REGION_LOW = np.array([CSR[0][0] - CSR[1][0], CSR[0][1] - CSR[1][1], CSR[0][2] - CSR[1][2]])
        self.CUBE_START_REGION_HIGH = np.array([CSR[0][0] + CSR[1][0], CSR[0][1] + CSR[1][1], CSR[0][2] + CSR[1][2]])
        # cube region (all places the cube could possibly exist at)
        CR = self.CUBE_REGION
        self.CUBE_REGION_LOW = np.array([CR[0][0] - CR[1][0], CR[0][1] - CR[1][1], CR[0][2] - CR[1][2]])
        self.CUBE_REGION_HIGH = np.array([CR[0][0] + CR[1][0], CR[0][1] + CR[1][1], CR[0][2] + CR[1][2]])
        # end effector possible position
        self.END_EFFECTOR_REGION_LOW = np.array([TR[0][0] - TR[1][0], TR[0][1] - TR[1][1], TR[0][2] - MAX_EF_HEIGHT])
        self.END_EFFECTOR_REGION_HIGH = np.array([TR[0][0] + TR[1][0], TR[0][1] + TR[1][1], TR[0][2] + MAX_EF_HEIGHT])

        # observations are dictionaries with the robot's joint angles and end effector position and
        # the cube's location and target location
        # each joint is between the min and max positions for the joint
        # the positions are a region in space encoded as the min(x,y,z) and max(x,y,z) for the box's region
        self.holding_cube = False
        self.observation_space = spaces.Dict(
            {
                "joints": spaces.Box(-np.pi*2, np.pi*2, shape=(6,), dtype=np.float32),  # ToDo: update limits according to getJointInfo() values
                "end_effector_pos": spaces.Box(self.TARGET_REGION_LOW, self.END_EFFECTOR_REGION_HIGH, shape=(3,), dtype=np.float32),
                "cube_pos": spaces.Box(self.CUBE_REGION_LOW, self.CUBE_REGION_HIGH, shape=(3,), dtype=np.float32),
                "target_pos": spaces.Box(self.TARGET_REGION_LOW, self.TARGET_REGION_HIGH, shape=(3,), dtype=np.float32)
            }
        )

        # we have 6 actions, corresponding to the angles of the six joints
        # ToDo: are we controlling velocity or position or torque of the joints?
        self.action_space = spaces.Box(-MAX_JOINT_VEL, MAX_JOINT_VEL, shape=(6,), dtype=np.float32)

        #self._action_to_angle_cmd = {
        #    0: np.array([1, 0]),
        #    1: np.array([0, 1]),
        #}

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode  # human or rgb_array
        self.verbose = verbose
        self.verbose_text = ""
        self.verbose_file = f"models/verbose/{int(time.time())}-{fname_app}.txt"
        self.resets = 0
        self.cur_steps = 0
        self.steps = 0
        self.score = 0

        self.total_steps = total_steps if total_steps is not None else 0
        self.start_time = time.time()

        self.physics_client = None
        self.cube_id = None
        self.robot_id = None
        self.joints_count = None
        self._end_effector_pos = None
        self._target_location = None
        self.prev_dist = self.dist = 0.09
        self.cube_constraint_id = None

    def set_fname(self, fname):
        self.fname = fname
        self.verbose_file = f"models/verbose/{self.fname}.txt"

    def print_verbose(self, s):
        if self.verbose:
            self.verbose_text += s + '\n'
            print(s)

    def _get_info(self):
        return {"step": self.steps, "cur_steps": self.cur_steps, "distance": np.linalg.norm(self._target_location, ord=1)}

    def _update_dist(self):
        self.prev_dist = self.dist
        if self.holding_cube:  # distance from ef to target_location
            self.dist = abs(np.linalg.norm(self._target_location - self._end_effector_pos)) - CUBE_DIM
        else:  # distance from ef to cube
            self.dist = abs(np.linalg.norm(self.cube_pos - self._end_effector_pos))

    def _get_centre_pos(self, region_low, region_high):
        pos = np.array([(region_low[0] + region_high[0]) / 2,
                        (region_low[1] + region_high[1]) / 2,
                        (region_low[2] + region_high[2]) / 2])
        return pos

    def _get_rnd_pos(self, region_low, region_high):
        pos = np.array([self.np_random.uniform(region_low[0], region_high[0]),
                        self.np_random.uniform(region_low[1], region_high[1]),
                        self.np_random.uniform(region_low[2], region_high[2])])
        return pos

    def _get_reward(self):
        #dist_to_reward = {0.05: 21, 0.15: 13, }
        # reward closer distance to cube (proportional reward)
        #reward = 1 / max(self.dist**2, 0.05**2)
        reward = 1 / max(self.dist, 0.05 / 2)
        # reward moving closer to cube than previous timestep
        #reward = 1.5 if self.dist < self.prev_dist else -1.5
        #print("dist:", self.dist, " reward:", reward)
        self.score += reward
        return reward

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.physics_client is None:
            self._setup()

        if self.render_mode == "human":
            time.sleep(1.0/240.0)

    def _get_obs(self):
        #joints_info = []
        #_jointIndexIdx = 0
        #_jointNameIdx = 1
        #_parentFramePosIdx = 14
        #joints_info = [pybullet.getJointInfo(self.robot_id, joint_num)[_parentFramePosIdx] for joint_num in range(joints_count)]

        # observations
        joint_positions = np.array([info[0] for info in pybullet.getJointStates(
            self.robot_id, range(0, 0+self.joints_count-1))], dtype=np.float32)
        self.cube_pos, cube_orn = pybullet.getBasePositionAndOrientation(self.cube_id)

        observations = {
            "joints": joint_positions,
            "end_effector_pos": self._end_effector_pos,
            "cube_pos": np.array(self.cube_pos, dtype=np.float32),
            "target_pos": np.array(self._target_location, dtype=np.float32)
        }

        return observations

    def step(self, action):
        # move joints
        for joint_index in range(0, 0+self.joints_count-1):
            pybullet.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=joint_index,
                                           controlMode=pybullet.VELOCITY_CONTROL, targetVelocity=action[joint_index])

        pybullet.stepSimulation()

        # ToDo: below is wrong, this is the joint position, but it should be the end effector position
        #self._end_effector_pos = np.array(pybullet.getLinkState(self.robot_id, 5+self.joints_count-1)[0], dtype=np.float32)
        self._end_effector_pos = np.array(pybullet.getLinkState(self.robot_id, self.joints_count-1)[0], dtype=np.float32)
        self._update_dist()

        print(f"ef pos: {pybullet.getLinkState(self.robot_id, self.joints_count-1)[0]}, dist: {self.dist}")

        if self.dist < CUBE_DIM and not self.holding_cube:
            self.holding_cube = True
            self.cube_constraint_id = pybullet.createConstraint(self.robot_id, self.joints_count-1,
                                                                self.cube_id, -1, pybullet.JOINT_FIXED,
                                                                [0, 0, 0], [0, 0, 0], [-(CUBE_DIM/2), 0, 0])

        terminated = False #self.dist < 0.056
        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()

        if terminated:
            reward += 5000
            self.score += reward

        if self.render_mode == "human":
            self._render_frame()

        self.steps += 1
        self.cur_steps += 1
        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        # we need the following line to seed self.np_random
        super().reset(seed=seed)

        # set up the environment if not yet done
        if self.physics_client is None:
            self._setup()

        # time remaining info
        time_remaining = 0
        if self.total_steps > 0:
            time_elapsed = time.time() - self.start_time
            steps_remaining = self.total_steps - self.steps
            time_remaining = int(steps_remaining * (time_elapsed / max(1, self.steps)))

        # verbose prints
        elapsed = int(time.time() - self.start_time)
        self.print_verbose(f"ETA: {time_remaining}s, total_steps: {self.steps}, sim: {self.resets}, "
                           f"steps: {self.cur_steps}, dist: {self.dist}, score: {self.score}, elapsed: {elapsed}s")

        # reset the robot's position
        pybullet.restoreState(self.init_state)

        # remove the cube to EF constraint
        self.holding_cube = False
        self.prev_dist = self.dist = 0.5
        if self.cube_constraint_id is not None:
            pybullet.removeConstraint(self.cube_constraint_id)

        # set a random position for the cube
        #cube_pos = self._get_rnd_pos(self.CUBE_START_REGION_LOW, self.CUBE_START_REGION_HIGH)
        self.cube_pos = self._get_centre_pos(self.CUBE_START_REGION_LOW, self.CUBE_START_REGION_HIGH)
        #cube_pos = np.array((self.CUBE_START_REGION[0]))
        cube_orn = pybullet.getQuaternionFromEuler([0, 0, 0])
        pybullet.resetBasePositionAndOrientation(self.cube_id, self.cube_pos, cube_orn)
        #pybullet.resetBasePositionAndOrientation(self.ef_cube_id, self._end_effector_pos, cube_orn)

        # set a random position for the target location
        #self._target_location = self._get_rnd_pos(self.TARGET_REGION_LOW, self.TARGET_REGION_HIGH)
        self._target_location = np.array((self.TARGET_REGION[0]))
        #self._end_effector_pos = np.array(pybullet.getLinkState(self.robot_id, 5+self.joints_count-1)[0], dtype=np.float32)
        self._end_effector_pos = np.array(pybullet.getLinkState(self.robot_id, self.joints_count-1)[0], dtype=np.float32)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.resets += 1
        self.cur_steps = 0
        self.score = 0

        return observation, info

    def _setup(self):
        # pybullet setup
        self.physics_client = pybullet.connect(pybullet.GUI if self.render_mode == "human" else pybullet.DIRECT)
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

            # cube to show where "EF" position is
            #self.ef_cube_id = pybullet.createVisualShape(shapeType=pybullet.GEOM_BOX,
            #                                             halfExtents=[0.05 / 2, 0.05 / 2, 0.05 / 2],
            #                                             visualFramePosition=[1.5, 1.5, 1.5],
            #                                             rgbaColor=[1, 1, 1, 1],
            #                                             specularColor=[0.4, .4, 0])
            #pybullet.createMultiBody(0, self.ef_cube_id, basePosition=[1.5, 1.5, 1.5])

            #self.ef_cube_id = pybullet.createVisualShape(ef_cube_shape_id,
            #                                             basePosition=self.CUBE_REGION[0])
            #basePosition=self.CUBE_REGION[0])  # [0.70, 0.0, 0.70])

        # box (an urdf file will not have accurate hit-boxes as it will fill in the empty space)
        mass = 0
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
        box_id = pybullet.createCollisionShapeArray(shapeTypes=shape_types,halfExtents=half_extents,
                                                    collisionFramePositions=collision_frame_positions)
        pybullet.createMultiBody(mass, box_id, basePosition=[0, 0, 0])

        # objects for pick-n-place
        cube_shape_id = pybullet.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=[0.05/2, 0.05/2, 0.05/2])
        mass = 0.03
        self.cube_id = pybullet.createMultiBody(mass, cube_shape_id, basePosition=self.CUBE_REGION[0])#[0.70, 0.0, 0.70])
        self.cube_pos, cube_orn = pybullet.getBasePositionAndOrientation(self.cube_id)
        #print("(0) cube_pos, cube_orn:", cube_pos, cube_orn)

        #for i in range(200):
            #cube_shape_id = pybullet.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=[0.05/2, 0.05/2, 0.05/2])
            #cube_id = pybullet.createMultiBody(mass, cube_shape_id, basePosition=self._get_rnd_pos(self.CUBE_START_REGION_LOW, self.CUBE_START_REGION_HIGH))
            #cube_id = pybullet.createMultiBody(mass, cube_shape_id, basePosition=self._get_rnd_pos(self.TARGET_REGION_LOW, self.TARGET_REGION_HIGH))

        # ABB IRB120
        # ToDo: add inertia to urdf file
        start_pos = [0, 0, 0.18/2 + 0.65]
        start_orientation = pybullet.getQuaternionFromEuler([0, np.pi/2, 0])
        urdf_path = "robo_ml_gym/models/irb120/irb120.urdf"
        if "robo_ml_gym" not in os.listdir():  # if cwd is 1 level up, then prepend gym-examples/ dir
            urdf_path = "robo_ml_gym/" + urdf_path
        self.robot_id = pybullet.loadURDF(urdf_path, basePosition=start_pos, baseOrientation=start_orientation,
                                          useFixedBase=1, flags=pybullet.URDF_MAINTAIN_LINK_ORDER)
        self.joints_count = pybullet.getNumJoints(self.robot_id)
        #self.joints_count = 1

        #print(pybullet.getJointInfo(self.robot_id, 0))
        # (0, b'joint_1', 0, 7, 6, 1, 0.0, 0.0, -2.87979, 2.87979, 0.0, 4.36332, b'link_1', (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), -1)

        # set the center of mass frame (loadURDF sets base link frame) start_pos/Orn
        #pybullet.resetBasePositionAndOrientation(self.robot_id, start_pos, start_orientation)

        # was used in first round of simulations:
        pybullet.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=1,
                                       controlMode=pybullet.POSITION_CONTROL, targetPosition=-0.9)

        # wait for the robot to reach its starting position before saving the state
        for i in range(500):
            pybullet.stepSimulation()

        self.init_state = pybullet.saveState()

    def close(self):
        if self.physics_client is not None:
            #robot_pos, robot_orn = pybullet.getBasePositionAndOrientation(self.robot_id)
            pybullet.disconnect()
        with open(self.verbose_file, 'w') as f:
            f.write(self.verbose_text)
