from datetime import datetime
import os
import re
import sys
import time
from stable_baselines3 import PPO, SAC, A2C, TD3, DDPG  # PPO, SAC, A2C, TD3, DDPG, HER-replay buffer
from stable_baselines3.common.logger import configure
# from stable_baselines3.common.utils import set_random_seed
import wandb
import config
# import custom_callback

WANDB_PROJECT = "robo-ml-gym"
ENV_ROBOWORLD = "robo_ml_gym:robo_ml_gym/RoboWorld-v0"
CONFIG_FILE = "configs/config.yml"
CONFIG_RND_FILE = "configs/config_rnd.yml"
MODELS_DIR = "models/"
LAST_MODEL_FILE = os.path.join(MODELS_DIR, ".last_model_name.txt")
os.makedirs(os.path.join(MODELS_DIR, "verbose"), exist_ok=True)
SCORES_FILE = "scores.txt"
start_time = int(time.time())

CONFIG = config.get_rnd_config(CONFIG_FILE, CONFIG_RND_FILE)
try:
    if os.environ["ROBO_GYM_VISUALISE"] == "1":
        CONFIG_FILE = "configs/config_visual.yml"
        CONFIG = config.get_config(CONFIG_FILE)
        print(CONFIG)
except KeyError:
    pass
# no random config parameters on local machine
if sys.platform == "win32":
    CONFIG = config.get_config(CONFIG_FILE)

GROUP_PREFIX = "NA"
if CONFIG["meta"].get("group", None) is not None:
    GROUP_PREFIX = CONFIG["meta"]["group"]
GROUP = GROUP_PREFIX
# silence logger
logger = configure()
logger.set_level(level=50)
# set all random seeds
# set_random_seed(123)
# TODO: save the random seed used


# disable wandb logging on my local Windows machine
if sys.platform == "win32":
    WANDB_PROJECT = "test-project"
    #os.environ["WANDB_MODE"] = "offline"


def init_wandb():
    global start_time
    start_time = int(time.time())
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged: robo-ml-gym
        project=WANDB_PROJECT,
        group=GROUP,
        # track hyperparameters and run metadata
        config=CONFIG
    )


def close_wandb():
    wandb.finish()


def get_model_name(model, total_time_steps, file_label):
    algo_name = str(model.__class__).split('.')[-1].strip("'>")
    time_s = datetime.now().strftime("%y%m%d_%H%M%S")
    name = f"{algo_name}-v{int(total_time_steps/1000)}k-{file_label}-{time_s}"
    path = os.path.join(MODELS_DIR, name)
    return name, path


def save_to_last_models(path):
    with open(LAST_MODEL_FILE, 'a') as f:
        f.write('\n'+path)


def save_model(env, model, model_filename):
    model.save(model_filename)
    time.sleep(0.1)
    save_to_last_models(model_filename)
    # ensure info logs (verbose) have the same name as the model
    env.unwrapped.set_fname(model_filename.strip(MODELS_DIR))


def get_statistics(env, model, path):
    info = env.unwrapped.info
    successes = info["success_tally"]
    fails = info["fail_tally"]
    fails = 1 if fails == 0 else fails
    runs = successes + fails
    success_rate = int(successes / runs * 100)
    avg_score = int(info["carry_over_score"] / runs)
    avg_dist = info["dist_tally"] / runs
    avg_ef_angle = int(info["ef_angle_tally"] / runs)
    total_steps = info["held_cube_step_tally"] + info["held_no_cube_step_tally"]
    held_cube_percent_time = info["held_cube_step_tally"] / total_steps * 100
    # avg_stack_dist is the avg of all (4) cubes from the stack position averaged across all time and sims
    avg_stack_dist = info["avg_stack_dist_tally"] / total_steps
    avg_cubes_stacked = info["cubes_stacked_tally"] / runs
    held_cube_rate = info["held_cube_tally"] / runs  # this is if a cube was held at end of sim
    avg_runtime = int((time.time() - start_time) / runs)

    avg_dist_str = "%.2f" % avg_dist
    avg_stack_dist_str = "%.2f" % avg_stack_dist
    avg_cubes_stacked_str = "%.2f" % avg_cubes_stacked
    held_cube_rate_str = "%.2f" % held_cube_rate
    info_str = (f"\n{path},{model.learning_rate},{avg_dist_str},{avg_stack_dist_str},{avg_ef_angle}," +
                f"{held_cube_rate_str},{avg_score},{successes},{fails},{success_rate},{avg_cubes_stacked_str}")

    results = {
                #"time:", learning_rate,
                #"ETA":
                "Avg Score": avg_score,
                "Successes": successes,
                "Fails": fails,
                "Success Rate": success_rate,
                #"ef_cube_dist": avg_ef_cube_dist,
                "Avg Distance": avg_dist,
                "Avg Stack Distance": avg_stack_dist,
                "Avg Cubes Stacked": avg_cubes_stacked,
                "Avg EF Angle": avg_ef_angle,
                "Held Cube Time Percentage": held_cube_percent_time,
                "Held Cube Rate": held_cube_rate,
                "Avg Runtime": avg_runtime
    }
    return results, info_str


def save_score(env, model, path, wandb_enabled, scores_path=SCORES_FILE):
    results, info_str = get_statistics(env, model, path)
    with open(scores_path, 'a') as f:
        f.write(info_str)
    if wandb_enabled:
        # TODO: record SB3 metrics: approx_kl, clip_fraction, entropy_loss etc
        wandb.log(results)


def get_previous_model_names():
    last_model_names = []
    with open(LAST_MODEL_FILE, 'r') as f:
        last_model_names = f.readlines()
    last_model_names.reverse()
    return last_model_names


def get_previous_model(env, custom_objects=None, match_str=None):
    if custom_objects is None:
        custom_objects = {}
    for last_model_name in get_previous_model_names():
        if match_str is not None:
            if match_str not in last_model_name:
                continue

        print("Loading model:", last_model_name)
        last_model_name = last_model_name.strip('\n')

        try:
            # get the number of steps previously taken while training this model
            prev_steps = int(re.search("-v([0-9]*)k-", last_model_name).group()[2:-2]) * 1_000

            # load and train model
            models_dict = {"PPO": PPO, "SAC": SAC, "A2C": A2C, "TD3": TD3, "DDPG": DDPG}
            for model_name, model_class in models_dict.items():
                if model_name in last_model_name:
                    model = PPO.load(last_model_name, env)
                    #model = model_class.load(last_model_name, env, custom_objects=custom_objects)
                    return model, prev_steps
        except Exception as exc:
            print(exc)
            pass

    return None, 0


def step_callback(model, _):
    try:
        _env = model["env"].envs[0].unwrapped
        print(_env.total_steps)
    except Exception as exc:
        print(exc)


def run(env, model, label, total_time_steps, prev_steps=0):
    #model = PPO("MultiInputPolicy", env, n_steps=20000, batch_size=128, n_epochs=20, verbose=1, learning_rate=0.0005, device="auto")  # promising
    name, path = get_model_name(model, prev_steps+total_time_steps, label)
    env.unwrapped.set_fname(name)  # ensure info logs have the same name as the model
    model.set_logger(logger)

    try:
        model.learn(total_timesteps=total_time_steps)#, callback=step_callback)
        save_score(env, model, path, True)
    except KeyboardInterrupt as exc:
        save_model(env, model, path)
        raise KeyboardInterrupt from exc

    save_model(env, model, path)

    if env is not None:
        env.close()
