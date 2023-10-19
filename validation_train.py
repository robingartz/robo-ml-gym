import gymnasium as gym

import time
from stable_baselines3 import PPO, SAC, A2C  # PPO, SAC, A2C, TD3, DDPG, HER-replay buffer

class Run:
    def __init__(self, total_time_steps=500_000, env=None, model=None):
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
        time_s = int(time.time())
        model_filename = f"models/{algo_name}-v{int(self.total_time_steps/1000)}k-{file_name_append}-{time_s}"

        # train model & save it
        self.start_time = time.time()
        model.learn(total_timesteps=self.total_time_steps)#, callback=self.model_callback)
        model.save(model_filename)

        if env is not None:
            env.close()

        print("=== done ===")

    def model_callback(self, state, _):
        #print(time.time() - self.start_time)
        #print(info)
        time_elapsed = time.time() - self.start_time
        steps_elapsed = state["steps"]
        steps_remaining = self.total_time_steps - steps_elapsed
        time_remaining = steps_remaining * (time_elapsed / steps_elapsed)
        print(f"time remaining: {time_remaining}")


def main():
    #Run()

    env_name = "robo_ml_gym:robo_ml_gym/RoboWorld-v0"
    policy_name = "MultiInputPolicy"
    models_dict = {"PPO": PPO, "SAC": SAC, "A2C": A2C}
    total_time_steps_dict = {"PPO": 250_000, "SAC": 5_000, "A2C": 250_000}
    max_ep_steps = int(240 * 1.5)

    # PPO
    model_name = "PPO"
    env = gym.make(env_name, max_episode_steps=max_ep_steps, verbose=True, total_steps=total_time_steps_dict[model_name])
    model = models_dict[model_name](policy_name, env, verbose=1, learning_rate=2e-4, device="auto")
    Run(total_time_steps=total_time_steps_dict[model_name], env=env, model=model)
    del env, model

    ## SAC
    #model_name = "SAC"
    #env = gym.make(env_name, max_episode_steps=max_ep_steps, verbose=True, total_steps=total_time_steps_dict[model_name])
    #model = models_dict[model_name](policy_name, env, verbose=1, device="auto")
    #Run(total_time_steps=total_time_steps_dict[model_name], env=env, model=model)
    #del env, model

    # A2C
    model_name = "A2C"
    env = gym.make(env_name, max_episode_steps=max_ep_steps, verbose=True, total_steps=total_time_steps_dict[model_name])
    model = models_dict[model_name](policy_name, env, n_steps=240*2, learning_rate=5e-4, verbose=1, device="auto")
    Run(total_time_steps=total_time_steps_dict[model_name], env=env, model=model)
    del env, model


if __name__ == '__main__':
    main()


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
        observation, info = env.reset()"""