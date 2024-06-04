
def _reward_func_none(dist):
    return 0


def _reward_func_12dist_add4(dist):
    return -12 * dist + 4


def _reward_func_minimise_xy_dist(dist):
    return 0


def _reward_func_1over_dist_add_dim(dist):
    return 0.1 / (dist + 0.05 / 2)


def _reward_func_max_12dist_add4_60dist_add5(dist):
    return max(-12 * dist + 4, -60 * dist + 5)


"""def reward_1div_max_dist_cube_dim(dist):
    return 1 / max(dist, 0.05 / 2)


def _get_simple_reward_normalised(env):
    reward_per_stacked_cube = 5
    norm_ef_cube_dist = env.ef_cube_dist / env.init_ef_cube_dist
    reward = 1 / max(norm_ef_cube_dist, 0.05 / 2) / 40
    norm_cube_stack_dist = env.cube_stack_dist / env.init_cube_stack_dist
    reward += 1 / max(norm_cube_stack_dist, 0.05 / 2) / 40
    reward += reward_per_stacked_cube * env.cubes_stacked
    return reward
"""


class Rewards:
    REWARD_FUNC_MAP = {
        "0": _reward_func_none,
        "-12 * self.dist + 4": _reward_func_12dist_add4,
        "minimise_xy_dist": _reward_func_minimise_xy_dist,
        "0.1 / (self.dist + 0.05 / 2)": _reward_func_1over_dist_add_dim,
        "max(-12 * self.dist + 4, -60 * self.dist + 5)": _reward_func_max_12dist_add4_60dist_add5
    }

    def __init__(self, config_reward):
        self.config_reward = config_reward
        try:
            self.dist_reward_func = self.REWARD_FUNC_MAP[config_reward["dist_reward_func"]]
        except KeyError as exc:
            raise KeyError("Invalid reward function selected") from exc

    def get_reward(self, dist, ef_pos, ef_angle, target_pos, held_cube, cubes_stacked, suction_on, cube_dim) -> float:
        """reward function: the closer the EF is to the target, the higher the reward"""
        # TODO: allow ef_angle to pickup cubes from the sides/while cube is at an angle

        reward = self.dist_reward_func(dist)

        if ef_pos[2] < 0:
            reward += self.config_reward["reward_for_ef_ground_col"]

        if held_cube is not None:
            # reward += (1 / max(cube_stack_dist, 0.05 / 2)) / 40
            reward += self.config_reward["reward_per_held_cube"]
            if held_cube.pos[2] < cube_dim / 2 - 0.0001:
                reward += self.config_reward["reward_for_cube_ground_col"]

        if ef_pos[2] < target_pos[2] - cube_dim / 4:
            reward += self.config_reward["reward_for_ef_below_target_z"]

        if (held_cube is None and not suction_on) or (held_cube is not None and suction_on):
            reward += self.config_reward["reward_suction_control_scale"]
        else:
            reward -= self.config_reward["reward_suction_control_scale"]

        # reward more vertical EF
        if self.config_reward["reward_ef_vertical"]:
            reward += max(0, (ef_angle - 90.0) / 90.0) * self.config_reward["reward_ef_vertical_scale"]

        reward += self.config_reward["reward_per_stacked_cube"] * cubes_stacked

        if held_cube is None:
            reward += self.config_reward["reward_for_suction_on_without_cube"]

        #if cubes_stacked == cube_count:
        #    ep_steps_remaining = ep_step_limit - ep_step
        #    max_reward_per_step = 1 + REWARD_FOR_HELD_CUBE + REWARD_FOR_EF_VERTICAL + REWARD_PER_STACKED_CUBE * cube_count
        #    reward = max_reward_per_step * ep_steps_remaining

        return reward
