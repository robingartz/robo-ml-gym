from robo_ml_gym.tasks import Task


class Arrange(Task):
    def __init__(self, env):
        super().__init__(env)

    def get_success_fail_tally(self):
        successes = self.env.arranged_cubes == self.env.cube_count
        fails = 1 - successes
        return successes, fails

    def get_termination(self):
        return False
