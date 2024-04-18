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
REL_MAX_DIS = 2.0
FLT_EPSILON = 0.0000001

class RoboWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 14}

    def __init__(self, render_mode=None, verbose=True, save_verbose=True, total_steps=None, fname_app="_", constant_cube_spawn=False):
        # box dimensions (the area the cube can spawn within)
        FLAT = 0.1  # ToDo: figure out what to do for the regions that are 2D...
        MAX_EF_HEIGHT = 0.6  # ToDo: this value is debatable, should it be enforced etc?
        MAX_JOINT_VEL = 1.0  # max joint velocity

        # defines a centre point (x,y,z) and the (w/2,l/2,h/2) for a cube region in space for the possible starting cube
        # position and target position
        self.REL_REGION_LOW = np.array([-REL_MAX_DIS, -REL_MAX_DIS, -REL_MAX_DIS])
        self.REL_REGION_HIGH = np.array([REL_MAX_DIS, REL_MAX_DIS, REL_MAX_DIS])
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
        self.reached_target_with_cube = False
        self.constant_cube_spawn = constant_cube_spawn
        #self.observation_space = spaces.Dict(
        #    {
        #        "joints": spaces.Box(-np.pi*2, np.pi*2, shape=(6,), dtype=np.float32),  # ToDo: update limits according to getJointInfo() values
        #        "end_effector_pos": spaces.Box(self.TARGET_REGION_LOW, self.END_EFFECTOR_REGION_HIGH, shape=(3,), dtype=np.float32),
        #        "cube_pos": spaces.Box(self.CUBE_REGION_LOW, self.CUBE_REGION_HIGH, shape=(3,), dtype=np.float32),
        #        "target_pos": spaces.Box(self.TARGET_REGION_LOW, self.TARGET_REGION_HIGH, shape=(3,), dtype=np.float32)
        #    }
        #)
        self.observation_space = spaces.Dict(
            {
                "joints": spaces.Box(-np.pi*2, np.pi*2, shape=(6,), dtype=np.float32),  # ToDo: update limits according to getJointInfo() values
                "rel_pos": spaces.Box(self.REL_REGION_LOW, self.REL_REGION_HIGH, shape=(3,), dtype=np.float32),
                "height": spaces.Box(0.0, 2.0, shape=(1,), dtype=np.float32)
            }
        )

        # we have 6 actions, corresponding to the angles of the six joints
        # ToDo: are we controlling velocity or position or torque of the joints?
        self.action_space = spaces.Box(-MAX_JOINT_VEL, MAX_JOINT_VEL, shape=(6,), dtype=np.float32)
        #self.action_space = spaces.Box(-np.pi*2, np.pi*2, shape=(6,), dtype=np.float32)

        # run physics sim for n steps, then give back control to agent
        #self.steps_between_interaction = 1

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode  # human or rgb_array
        self.verbose = verbose
        self.save_verbose = save_verbose
        self.verbose_text = ""
        self.v_txt = "_"
        self.verbose_file = f"models/verbose/{int(time.time())}-{fname_app}.txt"
        self.resets = 0
        self.cur_steps = 0
        self.steps = 0
        self.score = 0
        self.carry_over_score = 0
        self.carry_has_cube = 0
        self.carry_has_no_cube = 0

        self.total_steps = total_steps if total_steps is not None else 0
        self.start_time = time.time()

        self.physics_client = None
        self.cube_id = None
        self.robot_id = None
        self.joints_count = None
        self._end_effector_pos = None
        self.prev_end_effector_pos = None
        self.cube_pos = None
        self.use_phantom_cube = True
        self.point = None
        self.line_x = None
        self.line_y = None
        self.line_z = None
        self.target_pos = None
        self.prev_dist = self.dist = 0.09
        self.cube_constraint_id = None
        self.holding_cube = False
        self.just_picked_up_cube = False
        self.picked_up_cube_count = 0
        self.repeat_worst_performances = True
        self.prev_init_poses = []  # [(pos, score), ...]

    def set_fname(self, fname):
        self.fname = fname
        self.verbose_file = f"models/verbose/{self.fname}.txt"

    def print_verbose(self, s):
        if self.verbose:
            print(s)
        if self.save_verbose:
            with open(self.verbose_file, 'a') as f:
                f.write(s + '\n')

    def _get_info(self):
        return {"step": self.steps, "cur_steps": self.cur_steps}

    def _print_info(self):
        elapsed = int(time.time() - self.start_time)
        self.score = max(0, self.score)
        self.print_verbose(f"ETA: {self._get_time_remaining()}s, total_steps: {self.steps}, sim: {self.resets}, "
                           f"steps: {self.cur_steps}, dist: %.4f, score: %4d, elapsed: {elapsed}s, "
                           f"has cube: {self.holding_cube}" % (self.dist, int(self.score)))

    def _get_time_remaining(self):
        # time remaining info
        time_remaining = 0
        if self.total_steps > 0:
            time_elapsed = time.time() - self.start_time
            steps_remaining = self.total_steps - self.steps
            time_remaining = int(steps_remaining * (time_elapsed / max(1, self.steps)))
        return time_remaining

    def _update_dist(self):
        if self.use_phantom_cube:
            self.dist = abs(np.linalg.norm(self.target_pos - self._end_effector_pos))
        else:
            self.dist = abs(np.linalg.norm(self.cube_pos - self._end_effector_pos))# - (CUBE_DIM / 2)

    def _get_const_pos(self, region_low, region_high):
        pos = np.array([(region_low[0] + region_high[0]) / 2,
                        (region_low[1] + region_high[1]) / 2,
                        (region_low[2] + region_high[2]) / 2])
        pos = np.array([(region_low[0] + 0.03),
                        (region_low[1] + 0.03),
                        (region_low[2] + 0.03)])
        return pos

    def _get_rnd_pos(self, region_low, region_high):
        pos = np.array([self.np_random.uniform(region_low[0], region_high[0]),
                        self.np_random.uniform(region_low[1], region_high[1]),
                        self.np_random.uniform(region_low[2], region_high[2])])
        return pos

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.physics_client is None:
            self._setup()

        if self.render_mode == "human":
            time.sleep(1.0/240.0)

    def _get_obs(self):
        """returns the observation space: array of joint positions and the distance between EF and target"""

        # observations
        joint_positions = np.array([info[0] for info in pybullet.getJointStates(
            self.robot_id, range(0, 0+self.joints_count-1))], dtype=np.float32)
        self.cube_pos, cube_orn = pybullet.getBasePositionAndOrientation(self.cube_id)
        self.cube_pos = (self.cube_pos[0], self.cube_pos[1], self.cube_pos[2] + CUBE_DIM / 2)

        if self.use_phantom_cube:
            rel_pos = self._end_effector_pos - self.target_pos
        else:
            rel_pos = self._end_effector_pos - self.cube_pos
        np.clip(rel_pos, -REL_MAX_DIS, REL_MAX_DIS)
        rel_pos = rel_pos.astype("float32")
        height = np.array([min(max(self._end_effector_pos[2] - 0.0, 0.0), 2.0)], dtype=np.float32)
        observations = {
            "joints": joint_positions,
            "rel_pos": rel_pos,
            "height": height
        }

        return observations

    def _get_reward(self):
        """reward function: the closer the EF is to the target, the higher the reward"""
        self.normalise_by_init_dist = True
        # TODO: try normalise the reward by the starting distance
        # TODO: check that rel_pos is actually correct... when i move it around
        #reward = 1 / (self.dist * 20)
        reward = 1 / max(self.dist, 0.05 / 2)

        if self._end_effector_pos[2] < 0:
            reward -= 10

        #if self.holding_cube:
        #    reward += 2

        #if self.just_picked_up_cube and self.picked_up_cube_count == 1:
        #    reward += 50

        self.score += reward
        self.prev_dist = self.dist
        return reward

    def step(self, action):
        """move joints, step physics sim, check gripper, return obs, reward, termination"""
        # move joints
        for joint_index in range(0, 0+self.joints_count-1):
            pybullet.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=joint_index,
                                           controlMode=pybullet.POSITION_CONTROL, targetPosition=action[joint_index])
            #pybullet.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=joint_index,
            #                               controlMode=pybullet.VELOCITY_CONTROL, targetVelocity=action[joint_index])

        # how will this work in a visualisation? probably cannot just do 1 step as this is different to training
        #for i in range(self.steps_between_interaction):
        pybullet.stepSimulation()

        self.prev_end_effector_pos = self._end_effector_pos
        self._end_effector_pos = np.array(pybullet.getLinkState(self.robot_id, self.joints_count-1)[0], dtype=np.float32)
        self._update_dist()

        if self.holding_cube:
            self.just_picked_up_cube = False
        if self.dist < (CUBE_DIM / 2):  # * 1.45:
            if not self.holding_cube:
                print("creating cube constraint")
                self.holding_cube = True
                self.just_picked_up_cube = True
                self.picked_up_cube_count += 1
                self.cube_constraint_id = pybullet.createConstraint(self.robot_id, self.joints_count-1,
                                                                    self.cube_id, -1, pybullet.JOINT_FIXED,
                                                                    [0, 0, 0], [0, 0, 0], [-(CUBE_DIM/2), 0, 0])
        else:
            if self.holding_cube:
                pass
                #if self.dist > (CUBE_DIM / 2) * 20.5:
                #    self.holding_cube = False
                #    pybullet.removeConstraint(self.cube_constraint_id)
                #    self.cube_constraint_id = None

        terminated = False  #self.dist < 0.056
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
        """resets the robot position, cube position, score and gripper states"""
        # we need the following line to seed self.np_random
        super().reset(seed=seed)

        # set up the environment if not yet done
        if self.physics_client is None:
            self._setup()

        self._print_info()

        # reset the robot's position
        pybullet.restoreState(self.init_state)

        # remove the cube to EF constraint
        if self.holding_cube:
            self.carry_has_cube += 1
        else:
            self.carry_has_no_cube += 1
        self.holding_cube = False
        self.picked_up_cube_count = 0
        self.reached_target_with_cube = False
        self.prev_dist = self.dist = 0.5
        if self.cube_constraint_id is not None:
            pybullet.removeConstraint(self.cube_constraint_id)
            self.cube_constraint_id = None

        # set a random position for the cube
        if self.constant_cube_spawn:
            self.cube_pos = self._get_const_pos(self.CUBE_START_REGION_LOW, self.CUBE_START_REGION_HIGH)
        else:
            self.cube_pos = self._get_rnd_pos(self.CUBE_START_REGION_LOW, self.CUBE_START_REGION_HIGH)
        if self.use_phantom_cube:
            self.target_pos = self._get_rnd_pos(self.CUBE_START_REGION_LOW, self.CUBE_START_REGION_HIGH)
            #if self.line_x is not None:
            #    pybullet.removeUserDebugItem(self.line_x)
            #    pybullet.removeUserDebugItem(self.line_y)
            #    pybullet.removeUserDebugItem(self.line_z)
            #line_x_to = self.target_pos+[0.2, 0, 0]
            #line_y_to = self.target_pos+[0, 0.2, 0]
            #line_z_to = self.target_pos+[0, 0, 0.2]
            #self.line_x = pybullet.addUserDebugLine(lineFromXYZ=self.target_pos, lineToXYZ=line_x_to, lineColorRGB=[1, 0, 0], lifeTime=0)
            #self.line_y = pybullet.addUserDebugLine(lineFromXYZ=self.target_pos, lineToXYZ=line_y_to, lineColorRGB=[0, 1, 0], lifeTime=0)
            #self.line_z = pybullet.addUserDebugLine(lineFromXYZ=self.target_pos, lineToXYZ=line_z_to, lineColorRGB=[0, 0, 1], lifeTime=0)
            if self.point is not None:
                pybullet.removeUserDebugItem(self.point)
            self.point = pybullet.addUserDebugPoints(pointPositions=[self.target_pos], pointColorsRGB=[[0, 0, 1]], pointSize=10, lifeTime=1)

        # set cube orientation and position
        cube_orn = pybullet.getQuaternionFromEuler([0, 0, 0])
        pybullet.resetBasePositionAndOrientation(self.cube_id, self.cube_pos, cube_orn)

        self._end_effector_pos = np.array(pybullet.getLinkState(self.robot_id, self.joints_count-1)[0], dtype=np.float32)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.resets += 1
        self.cur_steps = 0
        self.carry_over_score += int(self.score)
        self.score = 0

        return observation, info

    def _setup(self):
        """pybullet setup"""
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
            pybullet.createMultiBody(0, plate_id, basePosition=[-0.01, 0.0, 0.45])

            # ABB circuit box
            abb_box_id = pybullet.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=[0.1, 0.2, 0.1])
            pybullet.createMultiBody(0, abb_box_id, basePosition=[0.1, 0.1, 0.1])

        # objects for pick-n-place
        cube_shape_id = pybullet.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=[0.05/2, 0.05/2, 0.05/2])
        mass = 0.03
        self.cube_id = pybullet.createMultiBody(1, cube_shape_id, basePosition=self.CUBE_REGION[0])#[0.70, 0.0, 0.70])
        self.cube_pos, cube_orn = pybullet.getBasePositionAndOrientation(self.cube_id)

        #for i in range(200):
        #    cube_shape_id = pybullet.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=[0.05/2, 0.05/2, 0.05/2])
        #    cube_id = pybullet.createMultiBody(mass, cube_shape_id, basePosition=self._get_rnd_pos(self.CUBE_START_REGION_LOW, self.CUBE_START_REGION_HIGH))
        #    cube_id = pybullet.createMultiBody(mass, cube_shape_id, basePosition=self._get_rnd_pos(self.TARGET_REGION_LOW, self.TARGET_REGION_HIGH))

        # ABB IRB120
        # ToDo: add inertia to urdf file
        start_pos = [0, 0, 0.18/2 + 0.65 - 0.1]
        start_orientation = pybullet.getQuaternionFromEuler([0, np.pi/2, 0])
        urdf_path = "robo_ml_gym/models/irb120/irb120.urdf"
        if "robo_ml_gym" not in os.listdir():  # if cwd is 1 level up, then prepend gym-examples/ dir
            urdf_path = "robo_ml_gym/" + urdf_path
        self.robot_id = pybullet.loadURDF(urdf_path, basePosition=start_pos, baseOrientation=start_orientation,
                                          useFixedBase=1, flags=pybullet.URDF_MAINTAIN_LINK_ORDER)
        self.joints_count = pybullet.getNumJoints(self.robot_id)

        # set the initial joint starting positions
        pybullet.resetJointState(bodyUniqueId=self.robot_id, jointIndex=1, targetValue=-0.9, targetVelocity=0)

        self.init_state = pybullet.saveState()

    def close(self):
        # unneeded as reset already saves to file
        #self._print_info()
        if self.physics_client is not None:
            pybullet.disconnect()
