import gymnasium as gym

from stable_baselines3 import A2C


class Run:
    def __init__(self):
        sims = 3000
        total_time_steps = 60_000

        env = gym.make("robo_ml_gym:robo_ml_gym/RoboWorld-v0", max_episode_steps=240*4, render_mode="human")

        # load models
        model = A2C.load("models/reach-model-A2C-v100k", env)
        #model = DQN.load("cpm/cartpole-model-dqn-v320k", env)
        #model = PPO.load("cpm/cartpole-model-ppo-v100k", env)

        vec_env = model.get_env()
        obs = vec_env.reset()
        score = 0
        for i in range(sims):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            score += reward
            vec_env.render("human")
            print(score)
            # VecEnv resets automatically
            # if done:
            #   obs = vec_env.reset()

        """
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
        """


if __name__ == '__main__':
    Run()
