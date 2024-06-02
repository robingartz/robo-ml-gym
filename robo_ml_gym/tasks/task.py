
class Task:
    def __init__(self, env):
        self.env = env

    def update_target_pos(self):
        ...

    def get_success_fail_tally(self):
        ...

    def get_termination(self):
        ...
