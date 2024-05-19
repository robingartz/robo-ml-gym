import gymnasium as gym
from stable_baselines3 import PPO, SAC, A2C  # PPO, SAC, A2C, TD3, DDPG, HER-replay buffer
import utils

GROUP_PREFIX = "A10"


def train_last_model(total_time_steps=100_000, max_episode_steps=240*4, constant_cube_spawn=False, learning_rate=5e-4):
    custom_objects = {'learning_rate': learning_rate}
    env = gym.make("robo_ml_gym:robo_ml_gym/RoboWorld-v0",
                   max_episode_steps=max_episode_steps,
                   ep_step_limit=max_episode_steps,
                   verbose=False,
                   total_steps_limit=total_time_steps,
                   constant_cube_spawn=constant_cube_spawn)

    model, prev_steps = utils.get_previous_model(env, custom_objects)

    # TODO: total_time_steps may be incorrect if training interrupted
    name, path = utils.get_model_name(model, prev_steps + total_time_steps, GROUP_PREFIX)
    env.unwrapped.set_fname(name)  # ensure info logs have the same name as the model
    utils.train_and_save_model(env, model, total_time_steps, path)

    if env is not None:
        env.close()


def run(env, model, total_time_steps, save_interval=0):
    #model = PPO("MultiInputPolicy", env, n_steps=20000, batch_size=128, n_epochs=20, verbose=1, learning_rate=0.0005, device="auto")  # promising

    name, path = utils.get_model_name(model=model, total_time_steps=total_time_steps, file_label=GROUP_PREFIX)
    utils.save_to_last_models(path)
    env.unwrapped.set_fname(name)  # ensure info logs have the same name as the model
    utils.train_and_save_model(env, model, total_time_steps, path)

    if env is not None:
        env.close()


def train_new_ppo(total_steps_limit=100_000, ep_step_limit=240*6, learning_rate=3e-4, batch_size=60, n_epochs=10):
    env_name = "robo_ml_gym:robo_ml_gym/RoboWorld-v0"
    env = gym.make(env_name, max_episode_steps=ep_step_limit, ep_step_limit=ep_step_limit,
                   verbose=False, total_steps_limit=total_steps_limit)
    model = PPO("MultiInputPolicy", env, learning_rate=learning_rate, verbose=1, device="auto", n_steps=ep_step_limit,
                batch_size=batch_size, n_epochs=n_epochs)
    run(total_time_steps=total_steps_limit, env=env, model=model)


def train_new_sac(total_steps_limit=20_000, ep_step_limit=240*6, learning_rate=3e-4, batch_size=60):
    env_name = "robo_ml_gym:robo_ml_gym/RoboWorld-v0"
    env = gym.make(env_name, max_episode_steps=ep_step_limit, ep_step_limit=ep_step_limit,
                   verbose=False, total_steps_limit=total_steps_limit)
    model = SAC("MultiInputPolicy", env, learning_rate=learning_rate, verbose=1, device="auto",
                batch_size=batch_size)
    run(total_time_steps=total_steps_limit, env=env, model=model)


if __name__ == '__main__':
    # probably worth having a few (8) points for the cube to spawn at and run those in batches?
    train_new_ppo(total_steps_limit=10_000)
    for i in range(20):
        train_last_model(total_time_steps=10_000, max_episode_steps=240*8, learning_rate=3e-4)
    for i in range(20):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=1e-4)
    for i in range(20):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=5e-5)
    for i in range(20):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=1e-5)
    for i in range(20):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=5e-6)
    for i in range(1000):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*12, learning_rate=1e-6)
