import gymnasium as gym


class Run:
    def __init__(self):
        env = gym.make("robo_bin_picking_gym:robo_bin_picking_gym/RoboWorld-v0", size=4, render_mode="human")

        observation, info = env.reset(seed=42)
        for i in range(100_000):
            action = env.action_space.sample()  # this is where you would insert your policy
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

        env.close()
