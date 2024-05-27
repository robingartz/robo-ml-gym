# standard libs
import time
import math
import os

# 3rd party libs
import numpy as np
import pybullet
import pybullet_data

# gym
import gymnasium as gym
from gymnasium import spaces

from robo_ml_gym.envs.cube import Cube
from robo_ml_gym.envs.region import Region


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
MAX_JOINT_VEL = 1.0
FLT_EPSILON = 0.0000001


class RoboWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 14}

    def __init__(self, render_mode=None, verbose=True, save_verbose=True, ep_step_limit=None,
                 total_steps_limit=None, fname_app="_", constant_cube_spawn=False, goal="phantom_touch",
                 orientation="vertical"):
        """
        PyBullet environment with the ABB IRB120 robot. The robot's end goal is
        to stack a number of cubes at the target_pos.

        :param render_mode:
        :param verbose:
        :param save_verbose:
        :param ep_step_limit:
        :param total_steps_limit:
        :param fname_app:
        :param constant_cube_spawn:
        :param goal: str "pickup" | "stack" | "touch" | "phantom_touch" | "place"
        :param orientation: str "vertical" | "horizontal"
        """

        # relative min/max
        self.REL_REGION_MIN = np.array([-REL_MAX_DIS, -REL_MAX_DIS, -REL_MAX_DIS])
        self.REL_REGION_MAX = np.array([REL_MAX_DIS, REL_MAX_DIS, REL_MAX_DIS])
        self.robot_workspace = Region([0.4, 0, 0.380])  # [0.6, 0, 0.580], [0.4, 0, 0.580], [0.4, 0, 0.380]

        # observations include joint angles, relative position between EF and target, and EF height
        # TODO: update joint limits according to getJointInfo() values or actual values
        self.observation_space = spaces.Dict(
            {
                "joints": spaces.Box(-np.pi*2, np.pi*2, shape=(6,), dtype=np.float32),
                "rel_pos": spaces.Box(self.REL_REGION_MIN, self.REL_REGION_MAX, shape=(3,), dtype=np.float32),
                "height": spaces.Box(0.0, 2.0, shape=(1,), dtype=np.float32)
            }
        )

        # we have 6 actions, corresponding to the angles of the six joints
        # TODO: are we controlling velocity or position or torque of the joints?
        self.action_space = spaces.Box(-MAX_JOINT_VEL, MAX_JOINT_VEL, shape=(6,), dtype=np.float32)
        #self.action_space = spaces.Box(-np.pi*2, np.pi*2, shape=(6,), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode  # human or rgb_array
        self.verbose = verbose
        self.visual_verbose = False
        self.save_verbose = save_verbose
        self.verbose_text = ""
        self.v_txt = "_"
        self.verbose_file = f"models/verbose/{int(time.time())}-{fname_app}.txt"

        # misc
        self.physics_client = None
        self.debug_points = []
        # run physics sim for n steps, then give back control to agent
        #self.total_steps_between_interaction = 1

        # used to estimate training ETA
        self.start_time = time.time()

        # step/reset counters
        self.resets = 0  # the number of resets; e.g. [0 to 100]
        self.ep_step = 0  # reset to 0 at the end of every episode; e.g. [0 to 240]
        self.ep_step_limit = ep_step_limit  # an episode will reset at this point; e.g. 240
        self.total_steps = 0  # the total number of steps taken in all episodes; e.g. [0 to 200_000]
        # total_steps_limit: training is completed once total_steps >= total_steps_limit; e.g. 200_000
        self.total_steps_limit = total_steps_limit if total_steps_limit is not None else 0

        # scoring
        self.goal = goal
        self.score = 0

        # robot vars
        self.orientation = orientation
        self.robot_id = None
        self.joints_count = None
        self.ef_pos = None  # end effector position (x, y, z)
        # 0 deg = vertical pointing up, 90 deg = horizontal EF, 180 deg = vertical pointing down
        self.ef_angle = 90
        self.ef_to_target_angle = 90
        self.home_pos = np.array([0.9, 0.0, 0.30])  # the home position for the robot once stacking is completed
        self.target_pos = None
        self.dist = 1.0
        self.ef_cube_dist = 1.0
        self.cube_stack_dist = 1.0

        self.init_ef_cube_dist = 1.0
        self.init_cube_stack_dist = 1.0

        # cube vars
        self.cube_count = 4
        self.cubes = []
        self.cube_ids = []
        self.stack_pos = None
        self.cubes_stacked = 0
        self.stack_tolerance = 0.05  # the tolerance allowed in the xy plane for stacked cubes to be considered stacked
        self.pickup_tolerance = 0.015
        self.pickup_xy_tolerance = 0.015
        self.cube_id = None  # TODO: remove use (only used once)
        self.cube_constraint_id = None
        self.held_cube = None
        self.picked_up_cube_count = 0
        self.orn_line = None  # debug line between EF and target

        # used from outer scope
        self.carry_over_score = 0
        self.carry_has_cube = 0
        self.carry_has_no_cube = 0
        self.success_tally = 0
        self.fail_tally = 0
        self.dist_tally = 0
        self.ef_angle_tally = 0

        # unused
        self.constant_cube_spawn = constant_cube_spawn
        self.prev_dist = self.dist
        self.prev_end_effector_pos = None
        self.just_picked_up_cube = False
        self.repeat_worst_performances = True

    def step(self, action):
        """move joints, step physics sim, check gripper, return obs, reward, termination"""
        self._process_action(action)

        # how will this work in a visualisation? probably cannot just do 1 step as this is different to training
        #for i in range(self.total_steps_between_interaction):
        pybullet.stepSimulation()

        for cube in self.cubes:
            cube.pos, cube.orn = pybullet.getBasePositionAndOrientation(cube.Id)
        self._check_cubes_stacked()
        unstacked_cube = self.get_first_unstacked_cube()  # TODO: this is the held cube

        self.prev_end_effector_pos = self.ef_pos
        ef_pos = pybullet.getLinkState(self.robot_id, self.joints_count-1)[0]
        self.ef_pos = np.array(ef_pos, dtype=np.float32)
        self._update_dist(unstacked_cube)
        self._update_target_pos(unstacked_cube)

        if self.render_mode == "human":
            self._process_keyboard_events()

            # debug line from ef_pos to target_pos
            if self.orn_line is not None:
                pybullet.removeUserDebugItem(self.orn_line)
            self.orn_line = pybullet.addUserDebugLine(ef_pos, self.target_pos)

        terminated = False #self.held_cube is not None #False  #self.dist < 0.056
        reward = self._get_reward()
        observation = self._get_obs()
        info = self._get_info()
        #reward = self._process_terminated_state(terminated, reward)

        if self.render_mode == "human":
            self._render_frame()

        self.total_steps += 1
        self.ep_step += 1

        if self.ep_step == self.ep_step_limit-1 or terminated:
            self._print_info()

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        """resets the robot position, cube position, score and gripper states"""
        # seeds self.np_random
        super().reset(seed=seed)

        # set up the environment if not yet done
        if self.physics_client is None:
            self._setup()

        self._tally_successes_fails()

        # reset vars
        self.held_cube = None
        self.picked_up_cube_count = 0
        self.prev_dist = self.dist = 1.0
        self.ef_to_target_angle = 90
        if self.cube_constraint_id is not None:
            pybullet.removeConstraint(self.cube_constraint_id)
            self.cube_constraint_id = None

        # reset the robot's position
        pybullet.restoreState(self.init_state)

        for debug_point in self.debug_points:
            pybullet.removeUserDebugItem(debug_point)

        # reset cube_pos, target_pos values
        self.ef_pos = np.array(pybullet.getLinkState(self.robot_id, self.joints_count-1)[0], dtype=np.float32)
        self._reset_target_pos()
        self._reset_cubes()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.resets += 1
        self.ep_step = 0
        self.carry_over_score += int(self.score)
        self.score = 0

        return observation, info

    def _setup(self):
        """pybullet setup"""
        self.physics_client = pybullet.connect(pybullet.GUI if self.render_mode == "human" else pybullet.DIRECT)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

        # gravity, ground, visual objects & debug points, cubes, ABB IRB120
        pybullet.setGravity(0, 0, -9.81)
        plane_id = pybullet.loadURDF("plane.urdf")
        self._setup_visual_objects()
        self._setup_cubes()
        self._setup_irb120()

        self.init_state = pybullet.saveState()

    def _get_obs(self):
        """returns the observation space: array of joint positions and the distance between EF and target"""
        # observations
        joint_positions = np.array([info[0] for info in pybullet.getJointStates(
            self.robot_id, range(0, 0+self.joints_count-1))], dtype=np.float32)
        pos = (self.target_pos[0], self.target_pos[1], self.target_pos[2] + CUBE_DIM / 2)

        rel_pos = self.ef_pos - pos
        np.clip(rel_pos, -REL_MAX_DIS, REL_MAX_DIS)
        rel_pos = rel_pos.astype("float32")
        height = np.array([min(max(self.ef_pos[2] - 0.0, 0.0), 2.0)], dtype=np.float32)
        #linear_vel, angular_vel = pybullet.getBaseVelocity(bodyUniqueId=self.held_cube.Id)
        #if abs(np.linalg.norm(np.array(linear_vel))) < 0.3:
        self.print_visual("target_pos: [%.2f %.2f %.2f], rel_pos: [%.3f %.3f %.3f], dist: %.2f, ef_z_angle: %2d, stacked: %d"
                          % (self.target_pos[0], self.target_pos[1], self.target_pos[2],
                             rel_pos[0], rel_pos[1], rel_pos[2], self.dist, self.ef_angle, self.cubes_stacked))

        observations = {
            "joints": joint_positions,
            "rel_pos": rel_pos,
            "height": height
        }
        # TODO: add "stacked_count" to obs

        return observations

    def _get_reward(self):
        """reward function: the closer the EF is to the target, the higher the reward"""
        # TODO: allow ef_angle to pickup cubes from the sides!!!
        PENALTY_FOR_EF_GROUND_COL = 1
        PENALTY_FOR_CUBE_GROUND_COL = 1
        PENALTY_FOR_BELOW_TARGET_Z = 0
        REWARD_FOR_HELD_CUBE = 2
        REWARD_PER_STACKED_CUBE = 0

        reward = self._get_dist_reward()

        if self.ef_pos[2] < 0:
            reward -= PENALTY_FOR_EF_GROUND_COL

        if self.held_cube is not None:
            #reward += (1 / max(self.cube_stack_dist, 0.05 / 2)) / 40
            reward += REWARD_FOR_HELD_CUBE
            #if self.held_cube.pos[2] < CUBE_DIM / 2 - 0.0001:
            #    reward -= PENALTY_FOR_CUBE_GROUND_COL

        if self.ef_pos[2] < self.target_pos[2] - CUBE_DIM / 4:
            reward -= PENALTY_FOR_BELOW_TARGET_Z

        # reward more vertical EF
        reward += max(0, (self.ef_angle - 90) / 90) * 15
        #reward += (self.ef_to_target_angle / 180) ** 2

        #reward += REWARD_PER_STACKED_CUBE * self.cubes_stacked

        #if self.cubes_stacked == self.cube_count:
        #    ep_steps_remaining = self.ep_step_limit - self.ep_step
        #    max_reward_per_step = 1 + REWARD_FOR_HELD_CUBE + REWARD_FOR_EF_VERTICAL + REWARD_PER_STACKED_CUBE * self.cube_count
        #    reward = max_reward_per_step * ep_steps_remaining

        #reward = self._get_dist_reward()
        self.score += reward
        self.prev_dist = self.dist
        return reward

    def set_fname(self, fname):
        self.fname = fname
        self.verbose_file = f"models/verbose/{self.fname}.txt"

    def print_verbose(self, s):
        if self.verbose:
            print(s)
        if self.save_verbose:
            with open(self.verbose_file, 'a') as f:
                f.write(s + '\n')

    def print_visual(self, _str):
        if self.render_mode == "human":
            if self.visual_verbose:
                print(_str)

    def _get_info(self):
        return {"step": self.total_steps, "ep_step": self.ep_step}

    def _print_info(self):
        elapsed = int(time.time() - self.start_time)
        self.score = max(0, self.score)
        #self.print_verbose(f"ETA: {self._get_time_remaining()}s, total_steps: {self.total_steps+1}, sim: {self.resets}, "
        #                   f"steps: {self.ep_step+1}, cube_dist: %.4f, score: %4d, elapsed: {elapsed}s, "
        #                   f"has cube: {self.held_cube is not None}, cubes_stacked: {self.cubes_stacked}, stack_dist: %.4f"
        #                   % (self.ef_cube_dist, int(self.score), self.cube_stack_dist))
        if self.resets == 1:
            self.print_verbose("t_rem,  steps , resets,epstep,ef_dist, score ,elapsed,cube,stacked,stack_dist,ef_angle")
        held_cube = 1 if self.held_cube is not None else 0
        self.print_verbose(f"%5d, %7d, %6d, %5d,  %.3f, %6d, %6d, %3d, %6d,     %.3f,     %3d"
                           % (self._get_time_remaining(), self.total_steps+1, self.resets, self.ep_step+1,
                              self.ef_cube_dist, int(self.score), elapsed, held_cube, self.cubes_stacked,
                              self.cube_stack_dist, self.ef_angle))

    def _get_time_remaining(self):
        # time remaining info
        time_remaining = 0
        if self.total_steps_limit > 0:
            time_elapsed = time.time() - self.start_time
            total_steps_remaining = self.total_steps_limit - self.total_steps
            time_remaining = int(total_steps_remaining * (time_elapsed / max(1, self.total_steps)))
        return time_remaining

    def vector_angle(self, a, b):
        return math.acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) * 180 / np.pi

    def _update_dist(self, cube):
        # TODO: need self.dist for reward func
        # distance between EF to cube and stack pos
        self.ef_cube_dist = 1.0
        self.cube_stack_dist = 1.0

        if self.cubes_stacked == self.cube_count and self.cube_count > 0:
            # all cubes are stacked
            cube = self.cubes[0]

        if cube is not None:
            # cube is held
            cube_pos = np.array(cube.pos)
            cube_pos[2] += CUBE_DIM / 2
            self.ef_cube_dist = abs(np.linalg.norm(cube_pos - self.ef_pos))
            self.cube_stack_dist = abs(np.linalg.norm(self.stack_pos - cube_pos))

        # TODO: FIX ME
        self.dist = self.ef_cube_dist
        if cube is not None:
            self.dist = self.cube_stack_dist

        if self.goal == "phantom_touch":
            self.ef_cube_dist = abs(np.linalg.norm(self.target_pos - self.ef_pos))
            self.dist = self.ef_cube_dist

        # angle between EF to target and the Z-axis
        vec = np.array(self.target_pos) - np.array(pybullet.getLinkState(self.robot_id, self.joints_count - 1)[0])
        self.ef_to_target_angle = self.vector_angle(vec, np.array([0.0, 0.0, 1.0]))
        # angle between EF and Z-axis
        ef_2 = np.array(pybullet.getLinkState(self.robot_id, self.joints_count - 2)[0])
        ef_1 = np.array(pybullet.getLinkState(self.robot_id, self.joints_count - 1)[0])
        self.ef_angle = self.vector_angle(ef_1 - ef_2, np.array([0.0, 0.0, 1.0]))

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.physics_client is None:
            self._setup()

        if self.render_mode == "human":
            time.sleep(1.0/240.0)

    def _get_dist_reward(self):
        reward = 1 / max(self.dist, 0.05 / 2)
        return reward

    def _get_simple_reward_normalised(self):
        REWARD_PER_STACKED_CUBE = 5
        norm_ef_cube_dist = self.ef_cube_dist / self.init_ef_cube_dist
        reward = 1 / max(norm_ef_cube_dist, 0.05 / 2) / 40
        norm_cube_stack_dist = self.cube_stack_dist / self.init_cube_stack_dist
        reward += 1 / max(norm_cube_stack_dist, 0.05 / 2) / 40
        reward += REWARD_PER_STACKED_CUBE * self.cubes_stacked
        return reward

    def _process_keyboard_events(self):
        """process keyboard event"""
        keys = pybullet.getKeyboardEvents()
        key_next = ord('n')
        key_verbose = ord('q')
        key_reset = ord('r')
        if key_next in keys and keys[key_next] & pybullet.KEY_WAS_TRIGGERED:
            self.reset()
        if key_verbose in keys and keys[key_verbose] & pybullet.KEY_WAS_TRIGGERED:
            self.visual_verbose = not self.visual_verbose
        if key_reset in keys and keys[key_reset] & pybullet.KEY_WAS_TRIGGERED:
            self._reset_robot_joint_values()

    def _is_ef_angle_vertical(self) -> bool:
        """check if the EF angle is close to vertical"""
        return self.ef_angle > 135
        #return self.ef_to_target_angle > 135  # 135 / 180 * np.pi = 2.356

    def _pickup_cube(self, cube):
        # TODO: allow picking up from an angle
        #pybullet.rayTest()
        #pybullet.getBasePositionAndOrientation()
        #pybullet.getQuaternionFromEuler()
        #pybullet.getAxisAngleFromQuaternion()
        if self.ef_cube_dist < self.pickup_tolerance:
            if self._xy_close(cube.pos, self.ef_pos, self.pickup_xy_tolerance):
                if self._is_ef_angle_vertical():
                    if self.goal == "pickup" or self.goal == "stack" or self.goal == "touch":
                        self.held_cube = cube
                        self.just_picked_up_cube = True
                        self.picked_up_cube_count += 1
                        self.cube_constraint_id = pybullet.createConstraint(
                            self.robot_id, self.joints_count-1, self.cube_id, -1,
                            pybullet.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [-(CUBE_DIM/2), 0, 0])

    def _release_cube(self):
        if self.cube_constraint_id is not None and self.held_cube is not None:
            linear_vel, angular_vel = pybullet.getBaseVelocity(bodyUniqueId=self.held_cube.Id)
            ef_speed = abs(np.linalg.norm(np.array(linear_vel)))
            if ef_speed < 0.01:
                # TODO: EF needs to go up / wait for cube to drop for a few ms before moving to next cube
                self.held_cube = None
                pybullet.removeConstraint(self.cube_constraint_id)
                self.cube_constraint_id = None
            else:
                self.print_visual(f"too fast: {abs(np.linalg.norm(np.array(linear_vel)))}")
                pass

    def _xy_close(self, arr1, arr2, a_tol: float):
        """check if x & y for arr1 & arr2 are within tolerance a_tol"""
        if abs(arr2[0] - arr1[0]) < a_tol:
            if abs(arr2[1] - arr1[1]) < a_tol:
                return True
        return False

    def _process_cube_interactions(self):
        """pickup cube if EF close"""
        if self.held_cube is not None:
            self.just_picked_up_cube = False

        if self.held_cube is None:
            # pickup cube and move towards the stack_pos
            self._pickup_cube(self.get_first_unstacked_cube())

    def _process_cube_interactions_pickup_drop(self):
        """pickup cube if EF close, drop cube if close to target"""
        if self.held_cube is not None:
            self.just_picked_up_cube = False

        if self.held_cube is None:
            # move towards the cube
            if self.cubes_stacked < self.cube_count:
                if self.held_cube is None:
                    # pickup cube and move towards the stack_pos
                    self._pickup_cube(self.get_first_unstacked_cube())
            else:
                # all cubes stacked
                pass
        else:
            if self._xy_close(self.held_cube.pos, self.stack_pos, self.stack_tolerance):
                # release cube and move towards the next cube
                self._release_cube()
            else:
                # not holding cube and not close to stack_pos: move towards stack_pos
                pass

    def get_first_unstacked_cube(self):
        """check how many cubes (must be consecutive cubes) are on stack_pos in the xy plane"""
        try:
            return self.cubes[self.cubes_stacked]
        except IndexError:
            return None

    def _check_cubes_stacked(self):
        self.cubes_stacked = 0
        for idx, cube in enumerate(self.cubes):
            if self._xy_close(cube.pos, self.stack_pos, self.stack_tolerance):
                self.cubes_stacked = idx + 1
            else:
                break

    def _process_action(self, action):
        """ move robot joints with either position control or velocity control """
        for joint_index in range(0, 0+self.joints_count-1):
            pybullet.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=joint_index,
                                           controlMode=pybullet.POSITION_CONTROL, targetPosition=action[joint_index])
            #pybullet.setJointMotorControl2(bodyUniqueId=self.robot_id, jointIndex=joint_index,
            #                               controlMode=pybullet.VELOCITY_CONTROL, targetVelocity=action[joint_index])

    def _process_terminated_state(self, terminated, reward):
        if terminated:
            if self.ep_step_limit is not None:
                reward += (1 / (0.05 / 2)) * 1.2 * 240 * (self.ep_step_limit - self.total_steps)
            else:
                reward = 69000
            self.score += reward
        #self.score = min(self.score, 69000)
        if terminated:
            self.score += 1000
        return reward

    def _update_target_pos(self, unstacked_cube):
        if self.goal == "pickup":
            self._process_cube_interactions()
            self.target_pos = np.array(self.cubes[0].pos)
            self.target_pos[2] += CUBE_DIM / 2

        elif self.goal == "touch":
            self._process_cube_interactions()
            self.target_pos = np.array(self.cubes[0].pos)
            self.target_pos[2] += CUBE_DIM / 2

        elif self.goal == "phantom_touch":
            # only updates on reset
            pass

        elif self.goal == "stack":
            self._process_cube_interactions_pickup_drop()
            # TODO: the arm is moving too much when it releases (particularly when human helps)

            if unstacked_cube is not None:
                self.cube_id = unstacked_cube.Id
                if self.held_cube is not None:
                    self.target_pos = np.array(self.stack_pos)
                    self.target_pos[2] += CUBE_DIM * self.cubes_stacked
                else:
                    self.target_pos = np.array(unstacked_cube.pos)
                    self.target_pos[2] += CUBE_DIM / 2
            else:
                self.target_pos = self.home_pos

    def _tally_successes_fails(self):
        # held cube tallies
        if self.held_cube is not None:
            self.carry_has_cube += 1
        else:
            self.carry_has_no_cube += 1

        self.dist_tally += self.dist
        self.ef_angle_tally += self.ef_angle

        # tally success/failures
        if self.goal == "pickup":
            if self.held_cube is not None: self.success_tally += 1
            else: self.fail_tally += 1
        elif self.goal == "touch":
            if self.held_cube is not None: self.success_tally += 1
            else: self.fail_tally += 1
        elif self.goal == "phantom_touch":
            if self.dist < self.pickup_tolerance: self.success_tally += 1
            else: self.fail_tally += 1
        elif self.goal == "stack":
            if self.cubes_stacked == self.cube_count: self.success_tally += 1
            else: self.fail_tally += 1

    def _reset_target_pos(self):
        if self.cube_count > 0:
            self.target_pos = self.cubes[0].pos
        if self.goal == "phantom_touch":
            self.target_pos = self.robot_workspace.get_rnd_point_bounded_z(CUBE_DIM / 2, CUBE_DIM * 6)
            if self.render_mode == "human":
                debug_point = pybullet.addUserDebugPoints(
                    pointPositions=[self.target_pos], pointColorsRGB=[[0, 0, 1]], pointSize=10, lifeTime=1)
                self.debug_points.append(debug_point)

    def _reset_cubes(self):
        """resets all cubes and debug related widgets"""
        # reset cube vars
        self.cubes_stacked = 0
        self.stack_pos = self.robot_workspace.get_rnd_plane_point(CUBE_DIM)
        self._setup_cube_positions()

        self._update_dist(self.get_first_unstacked_cube())
        self.init_ef_cube_dist = self.ef_cube_dist
        self.init_cube_stack_dist = self.cube_stack_dist

        # debug point at stacking target location
        debug_point = pybullet.addUserDebugPoints(
            pointPositions=[self.stack_pos], pointColorsRGB=[[1, 0, 0]], pointSize=8, lifeTime=1)
        self.debug_points.append(debug_point)

    def _setup_visual_objects(self):
        if self.render_mode == "human":
            ## baseplate
            #plate_id = pybullet.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=[0.01, 0.1, 0.45])
            #pybullet.createMultiBody(0, plate_id, basePosition=[-0.01, 0.0, 0.45])

            ## ABB circuit box
            #abb_box_id = pybullet.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=[0.1, 0.2, 0.1])
            #pybullet.createMultiBody(0, abb_box_id, basePosition=[0.1, 0.1, 0.1])

            # 8 corner points of region
            points = self.robot_workspace.get_corners()
            pybullet.addUserDebugPoints(pointPositions=points, pointColorsRGB=[(0, 0.5, 0.5)]*8, pointSize=5, lifeTime=0)

    def _setup_cube_positions(self):
        # reset all cube positions
        for cube in self.cubes:
            cube.pos = self.robot_workspace.get_rnd_plane_point(CUBE_DIM/2)
            self._check_cubes_stacked()
            while self.cubes_stacked > 0:
                cube.pos = self.robot_workspace.get_rnd_plane_point(CUBE_DIM/2)
                self._check_cubes_stacked()

            cube.orn = pybullet.getQuaternionFromEuler([0, 0, 0])
            pybullet.resetBasePositionAndOrientation(cube.Id, cube.pos, cube.orn)

    def _setup_cubes(self):
        cube_shape_id = pybullet.createCollisionShape(shapeType=pybullet.GEOM_BOX, halfExtents=[0.05/2, 0.05/2, 0.05/2])
        mass = 1.0
        for i in range(self.cube_count):
            cube_id = pybullet.createMultiBody(mass, cube_shape_id, basePosition=self.robot_workspace.get_rnd_plane_point(CUBE_DIM/2))
            cube_pos, cube_orn = pybullet.getBasePositionAndOrientation(cube_id)
            self.cubes.append(Cube(cube_id, cube_pos, cube_orn))
        if self.cube_count > 0:
            self.cube_id = self.cubes[0].Id
        self.stack_pos = self.robot_workspace.get_rnd_plane_point(CUBE_DIM)
        self._setup_cube_positions()

    def _setup_irb120(self):
        # ToDo: add inertia to urdf file
        if self.orientation == "horizontal":
            base_pos = [0, 0, 0.18/2 + 0.65 - 0.3]
            base_orn = pybullet.getQuaternionFromEuler([0, np.pi/2, 0])
        elif self.orientation == "vertical":
            base_pos = [0.3, 0, 0]
            base_orn = pybullet.getQuaternionFromEuler([0, 0, 0])
        else:
            raise RuntimeError(f"Invalid orientation: {self.orientation}")

        urdf_path = "robo_ml_gym/models/irb120/irb120.urdf"
        if "robo_ml_gym" not in os.listdir():  # if cwd is 1 level up, then prepend gym-examples/ dir
            urdf_path = "robo_ml_gym/" + urdf_path
        self.robot_id = pybullet.loadURDF(urdf_path, basePosition=base_pos, baseOrientation=base_orn,
                                          useFixedBase=1, flags=pybullet.URDF_MAINTAIN_LINK_ORDER)
        self.joints_count = pybullet.getNumJoints(self.robot_id)
        self._reset_robot_joint_values()

    def _reset_robot_joint_values(self):
        """ sets the initial joint starting positions """
        if self.orientation == "horizontal":
            joint_values = [0.0, -0.9, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif self.orientation == "vertical":
            joint_values = [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        else:
            raise RuntimeError(f"Invalid orientation: {self.orientation}")

        for idx, joint_value in enumerate(joint_values):
            pybullet.resetJointState(bodyUniqueId=self.robot_id, jointIndex=idx, targetValue=joint_value, targetVelocity=0)

    def close(self):
        if self.physics_client is not None:
            pybullet.disconnect()
