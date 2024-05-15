import numpy as np

class Region:
    """ A 3D region in space """
    def __init__(self, robot_pos):
        self.robot_pos = np.array(robot_pos)
        self.abs_min_xyz = self.robot_pos + np.array([0.200, -0.320, -0.380])
        self.abs_max_xyz = self.robot_pos + np.array([0.540, +0.320, +0.000])

    def is_point_within(self, arr: np.array):
        """ determines whether the given array is within the region """
        within = False
        if self.abs_min_xyz <= arr <= self.abs_max_xyz:
            within = True
        return within

    def get_corners(self):
        """ returns all 8 corners of the bounded region box """
        low = self.abs_min_xyz
        high = self.abs_max_xyz
        w = high[0] - low[0]
        l = high[1] - low[1]
        h = high[2] - low[2]
        points = ((low[0] + 0, low[1] + 0, low[2] + 0),
                  (low[0] + w, low[1] + 0, low[2] + 0),
                  (low[0] + 0, low[1] + l, low[2] + 0),
                  (low[0] + w, low[1] + l, low[2] + 0),
                  (low[0] + 0, low[1] + 0, low[2] + h),
                  (low[0] + w, low[1] + 0, low[2] + h),
                  (low[0] + 0, low[1] + l, low[2] + h),
                  (low[0] + w, low[1] + l, low[2] + h)
                  )
        return points

    def get_const_point(self, z: int = None) -> np.array:
        return (self.abs_min_xyz + self.abs_max_xyz) / 2

    def get_rnd_plane_point(self, z: float = 0.0) -> np.array:
        """ returns a random point within the region and on the plane at z """
        arr = self.get_rnd_point()
        arr[2] = self.abs_min_xyz[2] + z
        return arr

    def get_rnd_point(self) -> np.array:
        """ returns a random point within the region """
        point = None
        dist = 999
        #while dist > 0.66:
        x = np.random.uniform(self.abs_min_xyz[0], self.abs_max_xyz[0])
        y = np.random.uniform(self.abs_min_xyz[1], self.abs_max_xyz[1])
        z = np.random.uniform(self.abs_min_xyz[2], self.abs_max_xyz[2])
        point = np.array([x, y, z])
        dist = abs(np.linalg.norm(point - self.robot_pos))
        return point
