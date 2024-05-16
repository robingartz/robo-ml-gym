

class Cube:
    def __init__(self, cube_id, cube_pos, cube_orn):
        self.Id = cube_id
        self.pos = cube_pos
        self.orn = cube_orn

    def get_top_pos(self):
        temp = self.pos
        temp[2] += 0.025
        return temp

