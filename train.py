import gymnasium as gym
from stable_baselines3 import PPO, SAC, A2C  # PPO, SAC, A2C, TD3, DDPG, HER-replay buffer
import utils


def train_new_ppo(config: dict):
    env = gym.make(utils.ENV_ROBOWORLD, config=utils.CONFIG, max_episode_steps=config["ep_step_limit"],
                   verbose=False, wandb_enabled=True)
    model = PPO("MultiInputPolicy", env, learning_rate=float(config["learning_rate"]), verbose=1, device="auto",
                n_steps=config["ep_step_limit"], batch_size=config["batch_size"], n_epochs=config["n_epochs"],
                use_sde=config["use_sde"])
    utils.run(env=env, model=model, label=utils.GROUP_PREFIX, total_time_steps=config["total_steps_limit"], prev_steps=0)


def train_new_sac(config: dict):
    env = gym.make(utils.ENV_ROBOWORLD, config=utils.CONFIG, max_episode_steps=config["ep_step_limit"],
                   verbose=False, wandb_enabled=True)
    model = SAC("MultiInputPolicy", env, learning_rate=float(config["learning_rate"]), verbose=1, device="auto",
                batch_size=config["batch_size"], use_sde=config["use_sde"])
    utils.run(env=env, model=model, label=utils.GROUP_PREFIX, total_time_steps=config["total_steps_limit"], prev_steps=0)


def train_new_a2c(config: dict):
    env = gym.make(utils.ENV_ROBOWORLD, config=utils.CONFIG, max_episode_steps=config["ep_step_limit"],
                   verbose=False, wandb_enabled=True)
    model = A2C("MultiInputPolicy", env, learning_rate=float(config["learning_rate"]), verbose=1, device="auto",
                use_rms_prop=config["use_rms_prop"], use_sde=config["use_sde"])
    utils.run(env=env, model=model, label=utils.GROUP_PREFIX, total_time_steps=config["total_steps_limit"], prev_steps=0)


def train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=3e-4, match_str=None):
    custom_objects = {'learning_rate': learning_rate}
    env = gym.make(utils.ENV_ROBOWORLD,
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
        _learning_rate = float(utils.CONFIG["policy"]["learning_rate"])
        _learning_rate_5M = float(utils.CONFIG["policy"]["learning_rate_5M"])
        _learning_rate_10M = float(utils.CONFIG["policy"]["learning_rate_10M"])
        _learning_rate_15M = float(utils.CONFIG["policy"]["learning_rate_15M"])
        _learning_rate_20M = float(utils.CONFIG["policy"]["learning_rate_20M"])

        policy_methods = {"PPO": train_new_ppo, "SAC": train_new_sac, "A2C": train_new_a2c}
        policy = utils.CONFIG["policy"]["policy"]

        # train new model
        #policy_methods[policy](utils.CONFIG["policy"])
        #train_new_ppo(utils.CONFIG["policy"])

        for i in range(49): train_last_model(total_time_steps=_total_steps_limit, max_episode_steps=_ep_step_limit, learning_rate=_learning_rate)
        for i in range(50): train_last_model(total_time_steps=_total_steps_limit, max_episode_steps=_ep_step_limit, learning_rate=_learning_rate_5M)
        for i in range(50): train_last_model(total_time_steps=_total_steps_limit, max_episode_steps=_ep_step_limit, learning_rate=_learning_rate_10M)
        for i in range(50): train_last_model(total_time_steps=_total_steps_limit, max_episode_steps=_ep_step_limit, learning_rate=_learning_rate_15M)
        for i in range(50): train_last_model(total_time_steps=_total_steps_limit, max_episode_steps=_ep_step_limit, learning_rate=_learning_rate_20M)
        for i in range(50): train_last_model(total_time_steps=_total_steps_limit, max_episode_steps=_ep_step_limit, learning_rate=_learning_rate_20M)
        for i in range(50): train_last_model(total_time_steps=_total_steps_limit, max_episode_steps=_ep_step_limit, learning_rate=_learning_rate_20M)
        utils.close_wandb()
