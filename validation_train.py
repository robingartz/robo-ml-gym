import gymnasium as gym

from stable_baselines3 import A2C
from stable_baselines3 import DQN, PPO


class Run:
    def __init__(self):
        sims = 3000
        total_time_steps = 10_000

        env = gym.make("robo_ml_gym:robo_ml_gym/RoboWorld-v0", max_episode_steps=240*4, verbose=True)

        # create new models
        model = A2C("MultiInputPolicy", env, n_steps=3000, verbose=1)
        #model = PPO("MlpPolicy", env, verbose=1)

        # load models
        #model = A2C.load("models/reach-model-A2C-v60k", env)
        #model = DQN.load("cpm/cartpole-model-dqn-v320k", env)
        #model = PPO.load("cpm/cartpole-model-ppo-v100k", env)

        algo_name = str(model.__class__).split('.')[-1].strip("'>")
        model_filename = f"models/reach-model-{algo_name}-v{int(total_time_steps/1000)}k"

        # train model & save it
        model.learn(total_timesteps=total_time_steps)
        model.save(model_filename)

        #vec_env = model.get_env()
        #obs = vec_env.reset()
        #score = 0
        #for i in range(sims):
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

        env.close()


if __name__ == '__main__':
    Run()
