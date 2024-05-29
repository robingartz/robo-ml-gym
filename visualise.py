import gymnasium as gym
import utils


class Run:
    def __init__(self, constant_cube_spawn=False):
        self.constant_cube_spawn = constant_cube_spawn

    def run_with_model(self, last_model_name="", ep_step_limit=240*16):
        last = False
        if last_model_name == "" or last_model_name == "last":
            last = True
        sims = 100_000

        env = gym.make("robo_ml_gym:robo_ml_gym/RoboWorld-v0", config=utils.CONFIG,
                       max_episode_steps=ep_step_limit, render_mode="human", verbose=True, save_verbose=False)

        # load model
        model = None
        if last:
            model, _ = utils.get_previous_model(env)
            model.ep_step_limit = ep_step_limit
        else:
            pass

        print(model.policy)
        vec_env = model.get_env()
        obs = vec_env.reset()
        score = 0
        for i in range(sims):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            score += reward
            vec_env.render("human")

    def run_without_model(self):
        sims = 18
        ep_step_limit = 240 * 4
        env = gym.make("robo_ml_gym:robo_ml_gym/RoboWorld-v0", config=utils.CONFIG,
                       max_episode_steps=ep_step_limit, render_mode="human", save_verbose=False)

        observation, info = env.reset()
        for sim in range(sims):
            observation, info = env.reset()
            for step in range(ep_step_limit):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

        env.close()


if __name__ == '__main__':
    Run().run_with_model()
    #Run().run_without_model()
