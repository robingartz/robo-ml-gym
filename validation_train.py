import gymnasium as gym

from datetime import datetime
import time
from stable_baselines3 import PPO, SAC, A2C  # PPO, SAC, A2C, TD3, DDPG, HER-replay buffer


class Run:
    def __init__(self, total_time_steps=500_000, env=None, model=None):
        result = {"keyboard_interrupt": False}
        self.total_time_steps = total_time_steps
        file_name_append = "R2"

        #env = gym.make("robo_ml_gym:robo_ml_gym/RoboWorld-v0", max_episode_steps=240*2, verbose=True, total_steps=self.total_time_steps)

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
        with open("models/.last_model_name.txt", 'w') as f:
            f.write(model_filename)
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


class Manager:
    def __init__(self, model_types_to_run=("PPO", "SAC", "A2C"), do_short_time_steps=True, vary_max_steps=False,
                 vary_learning_rates=False, repeats=3):
        # selected run options
        self.model_types_to_run = model_types_to_run
        self.do_short_time_steps = do_short_time_steps
        self.vary_max_steps = vary_max_steps
        self.vary_learning_rates = vary_learning_rates
        self.repeats = repeats

        # configurations
        self.env_name = "robo_ml_gym:robo_ml_gym/RoboWorld-v0"
        self.policy_name = "MultiInputPolicy"
        self.models_dict = {"PPO": PPO, "SAC": SAC, "A2C": A2C}

        # total time steps
        self.total_time_steps_dict = {"PPO": 50_000, "SAC": 55_000, "A2C": 750_000}
        if self.do_short_time_steps:
            self.total_time_steps_dict = {"PPO": 2_000, "SAC": 700, "A2C": 2_000}

        # max episode steps
        self.max_ep_steps_list = [240 * 0.5, 240 * 1.0, 240 * 2.5, 240 * 3.5]
        if not self.vary_max_steps:
            self.max_ep_steps_list = [240]

        # learning rates
        self.lrs_dict = {"PPO": [0.0005, 0.0010],  # PPO  0.00009, 0.0001, 0.0003,
                         "SAC": [0.0005, 0.0010],  # SAC  0.00009, 0.0001, 0.0003,
                         "A2C": [0.0010, 0.0020]}  # A2C  0.00010, 0.0004, 0.0007,
        if not self.vary_learning_rates:
            self.lrs_dict = {"PPO": [5e-4], "SAC": [3e-4], "A2C": [7e-4]}

    def run(self):
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
            env = gym.make(self.env_name, max_episode_steps=self.max_ep_steps, verbose=True, total_steps=self.total_time_steps_dict[model_name])
            model = self.models_dict[model_name](self.policy_name, env, learning_rate=lr, verbose=1, device="auto")
            Run(total_time_steps=self.total_time_steps_dict[model_name], env=env, model=model)
            del env, model

        # SAC
        model_name = "SAC"
        if model_name in self.model_types_to_run:
            lr = self.lrs_dict[model_name][self.lr_i]
            env = gym.make(self.env_name, max_episode_steps=self.max_ep_steps, verbose=True, total_steps=self.total_time_steps_dict[model_name])
            model = self.models_dict[model_name](self.policy_name, env, learning_rate=lr, verbose=1, device="auto")
            Run(total_time_steps=self.total_time_steps_dict[model_name], env=env, model=model)
            del env, model

        # A2C
        model_name = "A2C"
        if model_name in self.model_types_to_run:
            lr = self.lrs_dict[model_name][self.lr_i]
            env = gym.make(self.env_name, max_episode_steps=self.max_ep_steps, verbose=True, total_steps=self.total_time_steps_dict[model_name])
            model = self.models_dict[model_name](self.policy_name, env, learning_rate=lr, n_steps=5, verbose=1, device="auto")
            Run(total_time_steps=self.total_time_steps_dict[model_name], env=env, model=model)
            del env, model


if __name__ == '__main__':
    #m = Manager(model_types_to_run=["PPO", "SAC", "A2C"], do_short_time_steps=True, vary_max_steps=True,
    #            vary_learning_rates=False, repeats=2)
    m = Manager(model_types_to_run=["PPO"], do_short_time_steps=False, vary_max_steps=False,
                vary_learning_rates=False, repeats=1)
    m.run()


"""
# create environment
env = gym.make("robo_ml_gym:robo_ml_gym/RoboWorld-v0", max_episode_steps=240 * 2, verbose=True,
               total_steps=self.total_time_steps)

# create new model
model = PPO("MultiInputPolicy", env, n_steps=20000, batch_size=128, n_epochs=20, verbose=1,
            learning_rate=0.0005)

# load models
model = PPO.load("models/PPO-v200k", env)

# train & save model
model.learn(total_timesteps=self.total_time_steps)
model.save(model_filename)
"""

# vec_env = model.get_env()
# obs = vec_env.reset()
# score = 0
# for i in range(sims):
#    action, _state = model.predict(obs, deterministic=True)
#    obs, reward, done, info = vec_env.step(action)
#    score += reward
#    vec_env.render("human")
#    print(score)
#    # VecEnv resets automatically
#    # if done:
#    #   obs = vec_env.reset()

"""observation, info = env.reset(seed=42)
for i in range(100):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()



#model_name = "SAC"
#env = gym.make(env_name, max_episode_steps=max_ep_steps, verbose=True, total_steps=total_time_steps_dict[model_name])
#model = SAC.load("models/SAC-v33k-R2-1697786639", env)
##model = models_dict[model_name](policy_name, env, verbose=1, device="auto")
#Run(total_time_steps=total_time_steps_dict[model_name], env=env, model=model)
#del env, model
#exit()

"""