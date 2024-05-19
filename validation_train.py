import gymnasium as gym

from datetime import datetime
import re
import time
from stable_baselines3 import PPO, SAC, A2C  # PPO, SAC, A2C, TD3, DDPG, HER-replay buffer

GROUP_PREFIX = "A9"


class Run:
    def __init__(self, total_time_steps=500_000, env=None, model=None, save_interval=0):
        result = {"keyboard_interrupt": False}
        self.total_time_steps = total_time_steps
        file_name_append = GROUP_PREFIX

        env = gym.make("robo_ml_gym:robo_ml_gym/RoboWorld-v0", max_episode_steps=240*2, ep_step_limit=240*2,verbose=True, total_steps_limit=self.total_time_steps)

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
            if save_interval == 0:
                model.learn(total_timesteps=self.total_time_steps)
            else:
                time_steps_completed = 0
                while time_steps_completed < self.total_time_steps:
                    model.learn(total_timesteps=save_interval)
                    model.save(model_filename + "-s" + str(int(time_steps_completed/1000)) + "k")
                    time_steps_completed += save_interval
        except KeyboardInterrupt:
            result["keyboard_interrupt"] = True

        # save model
        model.save(model_filename)

        if env is not None:
            env.close()

        if result["keyboard_interrupt"]:
            raise KeyboardInterrupt()


def get_model_name(model, total_time_steps, file_name_append=GROUP_PREFIX):
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


def get_previous_model_names():
    last_model_names = []
    with open("models/.last_model_name.txt", 'r') as f:
        last_model_names = f.readlines()
    last_model_names.reverse()
    return last_model_names


def get_previous_model(env, custom_objects):
    for last_model_name in get_previous_model_names():
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


def train_last_model(total_time_steps=30_000, max_episode_steps=240*4, constant_cube_spawn=False, learning_rate=5e-4):
    custom_objects = {'learning_rate': learning_rate}
    env = gym.make("robo_ml_gym:robo_ml_gym/RoboWorld-v0",
                   max_episode_steps=max_episode_steps,
                   ep_step_limit=max_episode_steps,
                   verbose=False,
                   total_steps_limit=total_time_steps,
                   constant_cube_spawn=constant_cube_spawn)

    model, prev_steps = get_previous_model(env, custom_objects)
    #model.learning_rate = 0.0

    # TODO: total_time_steps may be incorrect if training interrupted
    model_filename = get_model_name(model, prev_steps + total_time_steps)
    env.unwrapped.set_fname(model_filename.strip("models/"))  # ensure info logs have the same name as the model

    try:
        model.learn(total_timesteps=total_time_steps)
        with open("scores.txt", 'a') as f:
            successes = env.unwrapped.success_tally
            fails = env.unwrapped.fail_tally
            success_rate = int(successes / (successes + fails) * 100)
            avg_score = int(env.unwrapped.carry_over_score / (successes + fails))
            f.write(f"\n{model_filename},{learning_rate},{avg_score},{successes},{fails},{success_rate}" )
    except KeyboardInterrupt as err:
        print(err)

    save_model(env, model, model_filename)

    if env is not None:
        env.close()


class Manager:
    def __init__(self, model_types_to_run=("PPO", "SAC", "A2C"), do_short_time_steps=False, vary_max_steps=False,
                 vary_learning_rates=False, repeats=1, total_steps_limit=None, constant_cube_spawn=False):
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
        if total_steps_limit is not None:
            for key in self.total_time_steps_dict.keys():
                self.total_time_steps_dict[key] = total_steps_limit

        # max episode steps
        self.max_ep_steps_list = [240 * 0.5, 240 * 1.0, 240 * 2.5, 240 * 3.5]
        if not self.vary_max_steps:
            self.max_ep_steps_list = [240*4]
        self.max_ep_steps_list = [240 * 6]

        # learning rates
        self.lrs_dict = {"PPO": [0.0005, 0.0010],  # PPO  0.00009, 0.0001, 0.0003,
                         "SAC": [0.0002, 0.0005, 0.0010],  # SAC  0.00009, 0.0001, 0.0003,
                         "A2C": [0.0010, 0.0020]}  # A2C  0.00010, 0.0004, 0.0007,
        if not self.vary_learning_rates:
            self.lrs_dict = {"PPO": [5e-4], "SAC": [1e-4], "A2C": [7e-4]}
        #self.lrs_dict = {"PPO": [0.0], "SAC": [0.000001], "A2C": [0.0]}

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

            env = gym.make(self.env_name, max_episode_steps=self.max_ep_steps, ep_step_limit=self.max_ep_steps, verbose=False,
                           total_steps_limit=self.total_time_steps_dict[model_name],
                           constant_cube_spawn=self.constant_cube_spawn)
            #model = self.models_dict[model_name](self.policy_name, env, learning_rate=lr, verbose=1, device="auto",
            #                                     n_steps=240*12, batch_size=60, n_epochs=10)
            model = self.models_dict[model_name](self.policy_name, env, learning_rate=lr, verbose=1, device="auto",
                                                 n_steps=240*4)#, batch_size=60, n_epochs=10)
            # ensure info logs have the same name as the model
            env.unwrapped.set_fname(get_model_name(model, self.total_time_steps_dict[model_name]).strip("models/"))
            Run(total_time_steps=self.total_time_steps_dict[model_name], env=env, model=model)
            del env, model

        # SAC
        model_name = "SAC"
        if model_name in self.model_types_to_run:
            lr = self.lrs_dict[model_name][self.lr_i]
            env = gym.make(self.env_name, max_episode_steps=self.max_ep_steps, ep_step_limit=self.max_ep_steps,
                           verbose=True, total_steps_limit=self.total_time_steps_dict[model_name], constant_cube_spawn=self.constant_cube_spawn)
            model = self.models_dict[model_name](self.policy_name, env, learning_rate=lr, verbose=1, device="auto")
            # ensure info logs have the same name as the model
            env.unwrapped.set_fname(get_model_name(model, self.total_time_steps_dict[model_name]).strip("models/"))
            Run(total_time_steps=self.total_time_steps_dict[model_name], env=env, model=model)
            del env, model

        # A2C
        model_name = "A2C"
        if model_name in self.model_types_to_run:
            lr = self.lrs_dict[model_name][self.lr_i]
            env = gym.make(self.env_name, max_episode_steps=self.max_ep_steps, ep_step_limit=self.max_ep_steps,
                           verbose=True, total_steps_limit=self.total_time_steps_dict[model_name], constant_cube_spawn=self.constant_cube_spawn)
            model = self.models_dict[model_name](self.policy_name, env, learning_rate=lr, n_steps=5, verbose=1, device="auto")
            # ensure info logs have the same name as the model
            env.unwrapped.set_fname(get_model_name(model, self.total_time_steps_dict[model_name]).strip("models/"))
            Run(total_time_steps=self.total_time_steps_dict[model_name], env=env, model=model)
            del env, model

def newSAC():
    lrs = [
        #(240*4, 3e-4),
        (240*4, 1e-4),
        (240*6, 6e-5),
        (240*8, 3e-5),
        (240*8, 1e-5)]
    #m = Manager(model_types_to_run=["SAC"], total_steps_limit=20_000, constant_cube_spawn=False, vary_learning_rates=False)
    #m.run()
    for es_lr in lrs:
        ep_steps, lr = es_lr
        for i in range(3):
            train_last_model(total_time_steps=20_000, max_episode_steps=ep_steps, learning_rate=lr)
    for i in range(100):
        train_last_model(total_time_steps=20_000, max_episode_steps=240*8, learning_rate=1e-5)


def train_new_ppo(total_steps_limit=100_000, max_ep_steps=240*6, learning_rate=3e-4, batch_size=64, n_epochs=10):
    env_name = "robo_ml_gym:robo_ml_gym/RoboWorld-v0"
    env = gym.make(env_name, max_episode_steps=max_ep_steps, ep_step_limit=max_ep_steps,
                   verbose=False, total_steps_limit=total_steps_limit)
    model = PPO("MultiInputPolicy", env, learning_rate=learning_rate, verbose=1, device="auto", n_steps=max_ep_steps,
                batch_size=batch_size, n_epochs=n_epochs)
    env.unwrapped.set_fname(get_model_name(model, total_steps_limit).strip("models/"))
    Run(total_time_steps=total_steps_limit, env=env, model=model)


if __name__ == '__main__':
    # probably worth having a few (8) points for the cube to spawn at and run those in batches?

    #m = Manager(model_types_to_run=["PPO", "SAC", "A2C"], do_short_time_steps=True, vary_max_steps=True,
    #            vary_learning_rates=False, repeats=2)

    #m = Manager(model_types_to_run=["SAC"], total_steps_limit=23_000, constant_cube_spawn=False, vary_learning_rates=False)
    #m.run()
    #newSAC()

    # run previously trained model
    #train_last_model(total_time_steps=20_000, max_episode_steps=240)
    #for i in range(20):
    #    train_last_model(total_time_steps=20_000, max_episode_steps=240)
    #for i in range(10):
    #    train_last_model(total_time_steps=20_000, max_episode_steps=240*2)
    #for i in range(3):
    #    train_last_model(total_time_steps=20_000, max_episode_steps=240*4)

    #m = Manager(model_types_to_run=["PPO"], total_steps_limit=100_000, constant_cube_spawn=False, vary_learning_rates=False)
    #m.run()
    train_new_ppo(total_steps_limit=100_000)
    #train_last_model(total_time_steps=100_000, max_episode_steps=240*6, learning_rate=5e-6)
    for i in range(10):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*6, learning_rate=1e-3)
    for i in range(30):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*6, learning_rate=1e-3)
    for i in range(40):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=3e-4)
    for i in range(40):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=1e-4)
    for i in range(40):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=5e-5)
    for i in range(40):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=1e-5)
    for i in range(40):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=5e-6)
    for i in range(100):
        train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=1e-6)
    #for i in range(20):
    #    train_last_model(total_time_steps=100_000, max_episode_steps=240*7, learning_rate=3e-5)
    #for i in range(20):
    #    train_last_model(total_time_steps=100_000, max_episode_steps=240*7, learning_rate=2e-6)
    #for i in range(4000):
    #    train_last_model(total_time_steps=100_000, max_episode_steps=240*8, learning_rate=1e-6)

