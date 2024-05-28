import os
import gymnasium as gym
from stable_baselines3 import PPO, SAC, A2C  # PPO, SAC, A2C, TD3, DDPG, HER-replay buffer
import utils

GROUP_PREFIX = "A19"
ENV_ROBOWORLD = "robo_ml_gym:robo_ml_gym/RoboWorld-v0"
os.makedirs("models/verbose", exist_ok=True)


def train_new_ppo(total_steps_limit=100_000, ep_step_limit=240*8, learning_rate=3e-4, batch_size=60, n_epochs=10):
    env = gym.make(ENV_ROBOWORLD, max_episode_steps=ep_step_limit, ep_step_limit=ep_step_limit,
                   verbose=False, total_steps_limit=total_steps_limit, wandb_enabled=True)
    model = PPO("MultiInputPolicy", env, learning_rate=learning_rate, verbose=1, device="auto", n_steps=ep_step_limit,
                batch_size=batch_size, n_epochs=n_epochs)
    utils.run(env=env, model=model, label=GROUP_PREFIX, total_time_steps=total_steps_limit, prev_steps=0)



def train_new_sac(total_steps_limit=20_000, ep_step_limit=240*8, learning_rate=3e-4, batch_size=60):
    env = gym.make(ENV_ROBOWORLD, max_episode_steps=ep_step_limit, ep_step_limit=ep_step_limit,
                   verbose=False, total_steps_limit=total_steps_limit, wandb_enabled=True)
    model = SAC("MultiInputPolicy", env, learning_rate=learning_rate, verbose=1, device="auto",
                batch_size=batch_size)
    utils.run(env=env, model=model, label=GROUP_PREFIX, total_time_steps=total_steps_limit, prev_steps=0)


def train_last_model(total_time_steps=100_000, max_episode_steps=240*8, constant_cube_spawn=False, learning_rate=3e-4,
                     match_str=None):
    custom_objects = {'learning_rate': learning_rate}
    env = gym.make(ENV_ROBOWORLD,
                   max_episode_steps=max_episode_steps,
                   ep_step_limit=max_episode_steps,
                   verbose=False,
                   total_steps_limit=total_time_steps,
                   wandb_enabled=True)

    model, prev_steps = utils.get_previous_model(env, custom_objects, match_str)
    utils.run(env=env, model=model, label=GROUP_PREFIX, total_time_steps=total_time_steps, prev_steps=prev_steps)


if __name__ == '__main__':
    #for r in range(3):
    #    train_new_ppo(total_steps_limit=100_000, ep_step_limit=240*1)
    #    train_last_model(total_time_steps=100_000, max_episode_steps=240 * 6, learning_rate=3e-4)
    #    train_last_model(total_time_steps=100_000, max_episode_steps=240 * 6, learning_rate=3e-4)
    #for r in range(3):
    #    train_new_ppo(total_steps_limit=100_000, ep_step_limit=240*6)
    #    train_last_model(total_time_steps=100_000, max_episode_steps=240 * 6, learning_rate=3e-4)
    #    train_last_model(total_time_steps=100_000, max_episode_steps=240 * 6, learning_rate=3e-4)
    #model = "models/PPO-v9000k-A16-240526_064133"

    for r in range(5):
        utils.init_wandb()
        train_new_ppo(total_steps_limit=100_000, ep_step_limit=240 * 8)
        train_last_model(total_time_steps=100_000, max_episode_steps=240 * 8, learning_rate=3e-4)
        for i in range(200):
            train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=3e-4)
        for i in range(10):
            train_last_model(total_time_steps=100_000, max_episode_steps=240*12, learning_rate=3e-4)
        #for i in range(20): train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=1e-4)
        #for i in range(20): train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=5e-5)
        #for i in range(20): train_last_model(total_time_steps=200_000, max_episode_steps=240*12, learning_rate=1e-5)
        #for i in range(20): train_last_model(total_time_steps=200_000, max_episode_steps=240*12, learning_rate=5e-6)
        utils.close_wandb()
    for i in range(200):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*12, learning_rate=3e-5)
