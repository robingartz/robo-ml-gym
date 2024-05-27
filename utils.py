from datetime import datetime
import os
import re
import time
from stable_baselines3 import PPO, SAC, A2C  # PPO, SAC, A2C, TD3, DDPG, HER-replay buffer

MODELS_DIR = "models/"
LAST_MODEL_FILE = "models/.last_model_name.txt"
SCORES_FILE = "scores.txt"


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


def save_score(env, model, path, scores_path=SCORES_FILE):
    with open(scores_path, 'a') as f:
        successes = env.unwrapped.success_tally
        fails = env.unwrapped.fail_tally
        fails = 1 if fails == 0 else fails
        runs = successes + fails
        success_rate = int(successes / runs * 100)
        avg_score = int(env.unwrapped.carry_over_score / runs)
        avg_dist = "%.2f" % (env.unwrapped.dist_tally / runs)
        avg_ef_angle = int(env.unwrapped.ef_angle_tally / runs)
        f.write(f"\n{path},{model.learning_rate},{avg_score},{successes},{fails},{success_rate},{avg_dist},{avg_ef_angle}")


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
        save_score(env, model, path)
    except KeyboardInterrupt as exc:
        save_model(env, model, path)
        raise KeyboardInterrupt from exc

    save_model(env, model, path)

    if env is not None:
        env.close()
