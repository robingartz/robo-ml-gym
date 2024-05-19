import gymnasium as gym
from stable_baselines3 import PPO, SAC, A2C  # PPO, SAC, A2C, TD3, DDPG, HER-replay buffer
import utils

GROUP_PREFIX = "A10"
ENV_ROBOWORLD = "robo_ml_gym:robo_ml_gym/RoboWorld-v0"


def train_new_ppo(total_steps_limit=100_000, ep_step_limit=240*8, learning_rate=3e-4, batch_size=60, n_epochs=10):
    env = gym.make(ENV_ROBOWORLD, max_episode_steps=ep_step_limit, ep_step_limit=ep_step_limit,
                   verbose=False, total_steps_limit=total_steps_limit)
    model = PPO("MultiInputPolicy", env, learning_rate=learning_rate, verbose=1, device="auto", n_steps=ep_step_limit,
                batch_size=batch_size, n_epochs=n_epochs)
    utils.run(env=env, model=model, label=GROUP_PREFIX, total_time_steps=total_steps_limit, prev_steps=0)


def train_new_sac(total_steps_limit=20_000, ep_step_limit=240*8, learning_rate=3e-4, batch_size=60):
    env = gym.make(ENV_ROBOWORLD, max_episode_steps=ep_step_limit, ep_step_limit=ep_step_limit,
                   verbose=False, total_steps_limit=total_steps_limit)
    model = SAC("MultiInputPolicy", env, learning_rate=learning_rate, verbose=1, device="auto",
                batch_size=batch_size)
    utils.run(env=env, model=model, label=GROUP_PREFIX, total_time_steps=total_steps_limit, prev_steps=0)


def train_last_model(total_time_steps=100_000, max_episode_steps=240*4, constant_cube_spawn=False, learning_rate=3e-4):
    custom_objects = {'learning_rate': learning_rate}
    env = gym.make(ENV_ROBOWORLD,
                   max_episode_steps=max_episode_steps,
                   ep_step_limit=max_episode_steps,
                   verbose=False,
                   total_steps_limit=total_time_steps,
                   constant_cube_spawn=constant_cube_spawn)

    model, prev_steps = utils.get_previous_model(env, custom_objects)
    utils.run(env, model, GROUP_PREFIX, total_time_steps, prev_steps)


if __name__ == '__main__':
    # probably worth having a few (8) points for the cube to spawn at and run those in batches?
    train_new_ppo(total_steps_limit=10_000)
    for i in range(20):
        train_last_model(total_time_steps=10_000, max_episode_steps=240*4, learning_rate=3e-4)
    for i in range(20):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*4, learning_rate=1e-4)
    for i in range(20):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=5e-5)
    for i in range(20):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=1e-5)
    for i in range(20):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=5e-6)
    for i in range(1000):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*12, learning_rate=1e-6)
