from robo_ml_gym.tasks import Task


class Touch(Task):
    def __init__(self, env):
        super().__init__(env)

    def get_success_fail_tally(self):
        successes = self.env.held_cube is not None
        fails = 1 - successes
        return successes, fails

    def get_termination(self):
        return False
