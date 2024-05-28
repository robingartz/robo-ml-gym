from datetime import datetime
import os
import re
import time
from stable_baselines3 import PPO, SAC, A2C  # PPO, SAC, A2C, TD3, DDPG, HER-replay buffer
import wandb

MODELS_DIR = "models/"
LAST_MODEL_FILE = "models/.last_model_name.txt"
SCORES_FILE = "scores.txt"


def init_wandb():
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        # "robo-ml-gym",
        project="robo-ml-gym",

        # track hyperparameters and run metadata
        config={
            "policy": "PPO",
            "learning_rate": 3e-4,
            "learning_rates": "10M: 3e-4, 10M: 1e-4, 10M: 5e-5",
            "total_steps_limit": 100_000,
            "ep_step_limit": 240 * 8,
            "batch_size": 60,
            "n_epochs": 10,
            "robot_orientation": "vertical",
            "goal": "phantom_touch",
            "reward_func": "-12 * self.dist + 4",
            "obs_space": "joints (6), rel_pos (3), ef_height (1)",
            "action_space": "suction_on (1), joints (6)",
        }
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


def save_score(env, model, path, wandb_enabled, scores_path=SCORES_FILE):
    with open(scores_path, 'a') as f:
        info = env.unwrapped.info
        successes = info["success_tally"]
        fails = info["fail_tally"]
        fails = 1 if fails == 0 else fails
        runs = successes + fails
        success_rate = int(successes / runs * 100)
        avg_score = int(info["carry_over_score"] / runs)
        avg_dist = info["dist_tally"] / runs
        avg_dist_str = "%.2f" % avg_dist
        avg_ef_angle = int(info["ef_angle_tally"] / runs)
        held_cube_percent = info["held_cube_tally"] / (info["held_cube_tally"] + info["held_no_cube_tally"])
        f.write(f"\n{path},{model.learning_rate},{avg_score},{successes},{fails},{success_rate}," +
                f"{avg_dist_str},{avg_ef_angle}")

        if wandb_enabled:
            wandb.log(
                {
                    #"time:", learning_rate,
                    #"ETA":
                    "avg_score": avg_score,
                    "successes": successes,
                    "fails": fails,
                    "success_rate": success_rate,
                    #"ef_cube_dist": avg_ef_cube_dist,
                    #"cubes_stacked": avg_cubes_stacked,
                    "avg_dist": avg_dist,
                    "avg_ef_angle": avg_ef_angle,
                    "held_cube_percent": held_cube_percent
                })


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
            models_dict = {"PPO": PPO, "SAC": SAC, "A2C": A2C}
            for model_name, model_class in models_dict.items():
                if model_name in last_model_name:
                    model = model_class.load(last_model_name, env, custom_objects=custom_objects)
                    return model, prev_steps
        except Exception as error:
            print(error)
            pass

    return None, 0


def run(env, model, label, total_time_steps, prev_steps=0):
    #model = PPO("MultiInputPolicy", env, n_steps=20000, batch_size=128, n_epochs=20, verbose=1, learning_rate=0.0005, device="auto")  # promising
    name, path = get_model_name(model, prev_steps+total_time_steps, label)
    env.unwrapped.set_fname(name)  # ensure info logs have the same name as the model

    try:
        model.learn(total_timesteps=total_time_steps)
        save_score(env, model, path, True)
    except KeyboardInterrupt as exc:
        save_model(env, model, path)
        raise KeyboardInterrupt from exc

    save_model(env, model, path)

    if env is not None:
        env.close()
