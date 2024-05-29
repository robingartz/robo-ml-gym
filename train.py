import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO, SAC, A2C  # PPO, SAC, A2C, TD3, DDPG, HER-replay buffer
import utils

ENV_ROBOWORLD = "robo_ml_gym:robo_ml_gym/RoboWorld-v0"
os.makedirs("models/verbose", exist_ok=True)
# disable wandb logging on my local Windows machine
if sys.platform == "win32":
    os.environ["WANDB_MODE"] = "offline"


def train_new_ppo(total_steps_limit=100_000, ep_step_limit=240*8, learning_rate=3e-4, batch_size=60, n_epochs=10):
    env = gym.make(ENV_ROBOWORLD, config=utils.CONFIG, max_episode_steps=ep_step_limit, verbose=False, wandb_enabled=True)
    model = PPO("MultiInputPolicy", env, learning_rate=learning_rate, verbose=1, device="auto", n_steps=ep_step_limit,
                batch_size=batch_size, n_epochs=n_epochs)
    utils.run(env=env, model=model, label=utils.GROUP_PREFIX, total_time_steps=total_steps_limit, prev_steps=0)


def train_new_sac(total_steps_limit=20_000, ep_step_limit=240*8, learning_rate=3e-4, batch_size=60):
    env = gym.make(ENV_ROBOWORLD, config=utils.CONFIG, max_episode_steps=ep_step_limit, verbose=False, wandb_enabled=True)
    model = SAC("MultiInputPolicy", env, learning_rate=learning_rate, verbose=1, device="auto",
                batch_size=batch_size)
    utils.run(env=env, model=model, label=utils.GROUP_PREFIX, total_time_steps=total_steps_limit, prev_steps=0)


def train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=3e-4,
                     match_str=None):
    custom_objects = {'learning_rate': learning_rate}
    env = gym.make(ENV_ROBOWORLD,
                   config=utils.CONFIG,
                   max_episode_steps=max_episode_steps,
                   verbose=False,
                   wandb_enabled=True)

    model, prev_steps = utils.get_previous_model(env, custom_objects, match_str)
    utils.run(env=env, model=model, label=utils.GROUP_PREFIX, total_time_steps=total_time_steps, prev_steps=prev_steps)


if __name__ == '__main__':
    for r in range(5):
        utils.init_wandb()

        # get config vars
        _total_steps_limit = utils.CONFIG["policy"]["total_steps_limit"]
        _ep_step_limit = utils.CONFIG["policy"]["ep_step_limit"]
        _learning_rate = utils.CONFIG["policy"]["learning_rate"]
        _learning_rate_10M = utils.CONFIG["policy"]["learning_rate_10M"]
        _learning_rate_20M = utils.CONFIG["policy"]["learning_rate_20M"]

        train_new_ppo(total_steps_limit=_total_steps_limit, ep_step_limit=_ep_step_limit)
        for i in range(90): train_last_model(total_time_steps=_total_steps_limit, max_episode_steps=_ep_step_limit, learning_rate=_learning_rate)
        for i in range(100): train_last_model(total_time_steps=_total_steps_limit, max_episode_steps=_ep_step_limit, learning_rate=_learning_rate_10M)
        for i in range(100): train_last_model(total_time_steps=_total_steps_limit, max_episode_steps=_ep_step_limit, learning_rate=_learning_rate_20M)
        utils.close_wandb()
