import gymnasium as gym

from stable_baselines3 import A2C, PPO, SAC


def get_previous_model_names():
    last_model_names = []
    with open("models/.last_model_name.txt", 'r') as f:
        last_model_names = f.readlines()
    last_model_names.reverse()
    return last_model_names


def get_previous_model(env):
    for last_model_name in get_previous_model_names():
        print("Loading model:", last_model_name)
        last_model_name = last_model_name.strip('\n')

        try:
            # load model
            models_dict = {"PPO": PPO, "SAC": SAC, "A2C": A2C}
            for model_name, model_class in models_dict.items():
                if model_name in last_model_name:
                    model = model_class.load(last_model_name, env)
                    return model

        except Exception as error:
            print(error)
            pass

    return None


class Run:
    def __init__(self, constant_cube_spawn=False):
        self.constant_cube_spawn = constant_cube_spawn

    def run_with_model(self, last_model_name="", max_episode_steps=240*16):
        last = False
        if last_model_name == "" or last_model_name == "last":
            last = True
        sims = 190_000
        total_time_steps = 160_000

        env = gym.make("robo_ml_gym:robo_ml_gym/RoboWorld-v0", max_episode_steps=max_episode_steps, render_mode="human",
                       verbose=True, save_verbose=False, constant_cube_spawn=self.constant_cube_spawn)

        # load model
        model = None
        if last:
            model = get_previous_model(env)
            model.max_episode_steps = max_episode_steps
        else:
            #model = PPO.load("", env)
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
            #print(score)

    def run_without_model(self):
        sims = 18
        steps_per_sim = 240 * 4
        env = gym.make("robo_ml_gym:robo_ml_gym/RoboWorld-v0", max_episode_steps=steps_per_sim, render_mode="human",
                       save_verbose=False)

        observation, info = env.reset()#seed=42)
        for sim in range(sims):
            observation, info = env.reset()
            for step in range(steps_per_sim):
                action = env.action_space.sample()  # this is where you would insert your policy
                observation, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

        env.close()


if __name__ == '__main__':
    Run(constant_cube_spawn=False).run_with_model()
    #Run().run_without_model()
