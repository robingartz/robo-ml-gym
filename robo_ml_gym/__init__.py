from gymnasium.envs.registration import register

register(
     id="robo_ml_gym/RoboWorld-v0",
     entry_point="robo_ml_gym.envs:RoboWorldEnv",
     max_episode_steps=240*36,
)
