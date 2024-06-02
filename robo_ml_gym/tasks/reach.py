from robo_ml_gym.tasks import Task


class Reach(Task):
    def __init__(self, env):
        super().__init__(env)

    def get_success_fail_tally(self):
        successes = self.env.dist < self.env.pickup_tolerance
        fails = 1 - successes
        return successes, fails

    def get_termination(self):
        return False
