from gymnasium.envs.registration import register

register(
     id="robo_bin_picking_gym/RoboWorld-v0",
     entry_point="robo_bin_picking_gym.envs:RoboWorldEnv",
     max_episode_steps=300,
)
