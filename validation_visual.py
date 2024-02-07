import gymnasium as gym

from stable_baselines3 import A2C, PPO, SAC


class Run:
    def __init__(self):
        pass

    def run_with_model(self):
        sims = 190_000
        total_time_steps = 160_000

        env = gym.make("robo_ml_gym:robo_ml_gym/RoboWorld-v0", max_episode_steps=240*12, render_mode="human", verbose=True)

        # load models
        #model = PPO.load("models/PPO-v750k-R2-1697764483", env)
        #model = SAC.load("models/R3.1-vary-lr_ground/SAC-v55k-R2-1697792186", env)
        model = SAC.load("models/old/R3.0-vary-lr_ground/SAC-v55k-R2-1697734035", env)
        #model = A2C.load("models/R3-vary-lr/A2C-v750k-R2-1697750396", env)

        vec_env = model.get_env()
        obs = vec_env.reset()
        score = 0
        for i in range(sims):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            score += reward
            vec_env.render("human")
            #print(score)
            # VecEnv resets automatically
            # if done:
            #   obs = vec_env.reset()

    def run_without_model(self):
        sims = 5
        steps_per_sim = 240 * 4
        env = gym.make("robo_ml_gym:robo_ml_gym/RoboWorld-v0", max_episode_steps=steps_per_sim, render_mode="human")

        observation, info = env.reset(seed=42)
        for sim in range(sims):
            observation, info = env.reset()
            for step in range(steps_per_sim):
                action = env.action_space.sample()  # this is where you would insert your policy
                observation, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

        env.close()


if __name__ == '__main__':
    Run().run_with_model()
    #Run().run_without_model()
