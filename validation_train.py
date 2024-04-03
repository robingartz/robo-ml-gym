import gymnasium as gym

from datetime import datetime
import re
import time
from stable_baselines3 import PPO, SAC, A2C  # PPO, SAC, A2C, TD3, DDPG, HER-replay buffer


class Run:
    def __init__(self, total_time_steps=500_000, env=None, model=None):
        result = {"keyboard_interrupt": False}
        self.total_time_steps = total_time_steps
        file_name_append = "A2"

        env = gym.make("robo_ml_gym:robo_ml_gym/RoboWorld-v0", max_episode_steps=240*2, verbose=True, total_steps=self.total_time_steps)

        # create new models
        #model = PPO("MultiInputPolicy", env, n_steps=20000, batch_size=128, n_epochs=20, verbose=1, learning_rate=0.0005, device="auto")  # promising
        #model = SAC("MultiInputPolicy", env, verbose=1, device="auto")  # promising
        #model = A2C("MultiInputPolicy", env, n_steps=240*2, verbose=1, device="auto")  # promising  #, learning_rate=0.0004
        #model = TD3("MultiInputPolicy", env, verbose=1, device="auto")  # bad
        #model = DDPG("MultiInputPolicy", env, verbose=1, device="auto")  # very bad

        # load models
        #model = PPO.load("models/reach-model-PPO-v301k", env)
        #model = SAC.load("models/reach-model-SAC-v72k", env)
        #model = A2C.load("models/reach-model-A2C-v301k", env)
        #model = DQN.load("cpm/cartpole-model-dqn-v320k", env)

        algo_name = str(model.__class__).split('.')[-1].strip("'>")
        time_s = datetime.now().strftime("%y%m%d_%H%M%S")
        fname = f"{algo_name}-v{int(self.total_time_steps/1000)}k-{file_name_append}-{time_s}"
        model_filename = "models/" + fname
        with open("models/.last_model_name.txt", 'a') as f:
            f.write('\n' + model_filename)
        env.unwrapped.set_fname(fname)  # ensure info logs have the same name as the model

        # train model
        self.start_time = time.time()
        try:
            model.learn(total_timesteps=self.total_time_steps)#, callback=self.model_callback)
        except KeyboardInterrupt:
            result["keyboard_interrupt"] = True

        # save model
        model.save(model_filename)

        if env is not None:
            env.close()

        print("=== done ===")
        if result["keyboard_interrupt"]:
            raise KeyboardInterrupt()

    def model_callback(self, state, _):
        #print("ended - callback")
        #print(time.time() - self.start_time)
        #print(info)
        time_elapsed = time.time() - self.start_time
        #steps_elapsed = state["steps"]
        #steps_remaining = self.total_time_steps - steps_elapsed
        #time_remaining = steps_remaining * (time_elapsed / steps_elapsed)
        #print(f"time remaining: {time_remaining}")


def get_model_name(model, total_time_steps, file_name_append="A2"):
    algo_name = str(model.__class__).split('.')[-1].strip("'>")
    time_s = datetime.now().strftime("%y%m%d_%H%M%S")
    name = f"{algo_name}-v{int(total_time_steps/1000)}k-{file_name_append}-{time_s}"
    model_filename = "models/" + name
    return model_filename


def save_model_name(path):
    with open("models/.last_model_name.txt", 'a') as f:
        f.write('\n'+path)


def save_model(env, model, model_filename):
    model.save(model_filename)
    time.sleep(0.1)
    save_model_name(model_filename)
    # ensure info logs (verbose) have the same name as the model
    env.unwrapped.set_fname(model_filename.strip("models/"))


def train_last_model(total_time_steps=30_000, max_episode_steps=240*4, constant_cube_spawn=False):
    last = True
    model = None
    last_model_names = []
    with open("models/.last_model_name.txt", 'r') as f:
        last_model_names = f.readlines()

    env = gym.make("robo_ml_gym:robo_ml_gym/RoboWorld-v0",
                   max_episode_steps=max_episode_steps,
                   verbose=False,
                   total_steps=total_time_steps,
                   constant_cube_spawn=constant_cube_spawn)

    last_model_names.reverse()
    for last_model_name in last_model_names:
        print("Loading model:", last_model_name)
        last_model_name = last_model_name.strip('\n')

        try:
            # get the number of steps previously taken while training this model
            prev_steps = int(re.search("-v([0-9]*)k-", last_model_name).group()[2:-2]) * 1_000

            # load and train model
            model = PPO.load(last_model_name, env)
            break
        except:
            pass

    model_filename = get_model_name(model, prev_steps + total_time_steps)
    env.unwrapped.set_fname(model_filename.strip("models/"))  # ensure info logs have the same name as the model

    try:
        model.learn(total_timesteps=total_time_steps)
    except KeyboardInterrupt as err:
        print(err)

    # save model
    # TODO: total_time_steps may be incorrect if training interrupted
    model_filename = get_model_name(model, prev_steps + total_time_steps)
    save_model(env, model, model_filename)

    if env is not None:
        env.close()


class Manager:
    def __init__(self, model_types_to_run=("PPO", "SAC", "A2C"), do_short_time_steps=True, vary_max_steps=False,
                 vary_learning_rates=False, repeats=1, total_steps=None, constant_cube_spawn=False):
        # selected run options
        self.model_types_to_run = model_types_to_run
        self.do_short_time_steps = do_short_time_steps
        self.vary_max_steps = vary_max_steps
        self.vary_learning_rates = vary_learning_rates
        self.repeats = repeats
        self.constant_cube_spawn = constant_cube_spawn

        # configurations
        self.env_name = "robo_ml_gym:robo_ml_gym/RoboWorld-v0"
        self.policy_name = "MultiInputPolicy"
        self.models_dict = {"PPO": PPO, "SAC": SAC, "A2C": A2C}

        # total time steps
        self.total_time_steps_dict = {"PPO": 5_000, "SAC": 1_000, "A2C": 750_000}
        if self.do_short_time_steps:
            self.total_time_steps_dict = {"PPO": 6_000, "SAC": 700, "A2C": 2_000}
        if total_steps is not None:
            for key in self.total_time_steps_dict.keys():
                self.total_time_steps_dict[key] = total_steps

        # max episode steps
        self.max_ep_steps_list = [240 * 0.5, 240 * 1.0, 240 * 2.5, 240 * 3.5]
        if not self.vary_max_steps:
            self.max_ep_steps_list = [240*4]

        # learning rates
        self.lrs_dict = {"PPO": [0.0005, 0.0010],  # PPO  0.00009, 0.0001, 0.0003,
                         "SAC": [0.0005, 0.0010],  # SAC  0.00009, 0.0001, 0.0003,
                         "A2C": [0.0010, 0.0020]}  # A2C  0.00010, 0.0004, 0.0007,
        if not self.vary_learning_rates:
            self.lrs_dict = {"PPO": [5e-4], "SAC": [3e-4], "A2C": [7e-4]}

    def run(self):
        print(f"running training for: {self.model_types_to_run} for: {self.total_time_steps_dict} steps")
        try:
            for max_ep_steps in self.max_ep_steps_list:
                for self.lr_i in range(len(self.lrs_dict["PPO"])):
                    self.max_ep_steps = int(max_ep_steps)
                    for i in range(self.repeats):
                        self.run_models()

        except KeyboardInterrupt as err:
            print(err)

    def run_models(self):
        # PPO
        model_name = "PPO"
        if model_name in self.model_types_to_run:
            lr = self.lrs_dict[model_name][self.lr_i]
            #PPO(n_steps=2048, batch_size=64, n_epochs=10)

            env = gym.make(self.env_name, max_episode_steps=self.max_ep_steps, verbose=False,
                           total_steps=self.total_time_steps_dict[model_name],
                           constant_cube_spawn=self.constant_cube_spawn)
            model = self.models_dict[model_name](self.policy_name, env, learning_rate=lr, verbose=1, device="auto",
                                                 n_steps=240*12, batch_size=60, n_epochs=10)
            # ensure info logs have the same name as the model
            env.unwrapped.set_fname(get_model_name(model, self.total_time_steps_dict[model_name]).strip("models/"))
            Run(total_time_steps=self.total_time_steps_dict[model_name], env=env, model=model)
            del env, model

        # SAC
        model_name = "SAC"
        if model_name in self.model_types_to_run:
            lr = self.lrs_dict[model_name][self.lr_i]
            env = gym.make(self.env_name, max_episode_steps=self.max_ep_steps, verbose=True, total_steps=self.total_time_steps_dict[model_name], constant_cube_spawn=self.constant_cube_spawn)
            model = self.models_dict[model_name](self.policy_name, env, learning_rate=lr, verbose=1, device="auto")
            # ensure info logs have the same name as the model
            env.unwrapped.set_fname(get_model_name(model, self.total_time_steps_dict[model_name]).strip("models/"))
            Run(total_time_steps=self.total_time_steps_dict[model_name], env=env, model=model)
            del env, model

        # A2C
        model_name = "A2C"
        if model_name in self.model_types_to_run:
            lr = self.lrs_dict[model_name][self.lr_i]
            env = gym.make(self.env_name, max_episode_steps=self.max_ep_steps, verbose=True, total_steps=self.total_time_steps_dict[model_name], constant_cube_spawn=self.constant_cube_spawn)
            model = self.models_dict[model_name](self.policy_name, env, learning_rate=lr, n_steps=5, verbose=1, device="auto")
            # ensure info logs have the same name as the model
            env.unwrapped.set_fname(get_model_name(model, self.total_time_steps_dict[model_name]).strip("models/"))
            Run(total_time_steps=self.total_time_steps_dict[model_name], env=env, model=model)
            del env, model


if __name__ == '__main__':
    # probably worth having a few (8) points for the cube to spawn at and run those in batches?

    #m = Manager(model_types_to_run=["PPO", "SAC", "A2C"], do_short_time_steps=True, vary_max_steps=True,
    #            vary_learning_rates=False, repeats=2)

    #m = Manager(model_types_to_run=["PPO"], do_short_time_steps=False, total_steps=250_000, constant_cube_spawn=False)
    #m.run()

    # run previously trained model
    train_last_model(total_time_steps=2_000_000, max_episode_steps=240*4)
