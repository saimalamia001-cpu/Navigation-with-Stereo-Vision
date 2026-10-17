import numpy as np
from scipy.ndimage import distance_transform_edt
from .config import GRID, CAMERA, ROBOT
from .geometry import ray_cells


class OccupancyGrid:
    """Log-odds map. Unknown stays unknown when stereo has no evidence."""
    def __init__(self, config=GRID):
        self.config = config
        shape = (round(config.height_m / config.resolution), round(config.width_m / config.resolution))
        self.log_odds = np.zeros(shape, np.float32)
        self.observed = np.zeros(shape, bool)
        self.revision = 0

    def cell(self, xy):
        return np.floor(np.asarray(xy) / self.config.resolution).astype(int)

    def world(self, cell):
        return (np.asarray(cell) + 0.5) * self.config.resolution

    def inside(self, cell):
        x, y = cell
        return 0 <= x < self.log_odds.shape[1] and 0 <= y < self.log_odds.shape[0]

    @property
    def occupied(self):
        return self.log_odds > 0.7

    def update(self, pose, points, camera=CAMERA):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        origin = pose[:2] + camera.forward_m * np.array([c, s]) + camera.baseline / 2 * np.array([-s, c])
        # Points are measured relative to the LEFT lens.
        heights = points[:, 2] + camera.height_m
        usable = (heights > -0.06) & (heights < 1.4)
        p = points[usable]
        heights = heights[usable]
        ends = origin + np.column_stack((c * p[:, 0] - s * p[:, 1], s * p[:, 0] + c * p[:, 1]))
        free, hits = set(), set()
        start = self.cell(origin)
        # One nearest obstacle per bearing bin prevents free-space rays through walls.
        bearings = np.arctan2(p[:, 1], p[:, 0])
        bins = np.floor(bearings / 0.012).astype(int)
        for b in np.unique(bins):
            ids = np.flatnonzero(bins == b)
            obstacle_ids = ids[heights[ids] > 0.12]
            if len(obstacle_ids):
                # Require vertical support; a single outlier must not create a wall.
                ranges = p[obstacle_ids, 0]
                median = np.median(ranges)
                supported = obstacle_ids[np.abs(ranges - median) < .15]
                if len(supported) < 4 or np.ptp(heights[supported]) < .16:
                    continue
                i = supported[np.argmin(np.abs(p[supported, 0] - median))]
                hit = True
            else:
                i = ids[np.argmax(p[ids, 0])]
                hit = False
            end = self.cell(ends[i])
            cells = list(ray_cells(start, end))
            free.update(v for v in cells[:-1] if self.inside(v))
            if self.inside(end):
                (hits if hit else free).add(tuple(end))
        before = self.occupied.copy()
        for x, y in free - hits:
            self.log_odds[y, x] -= 0.35
            self.observed[y, x] = True
        for x, y in hits:
            self.log_odds[y, x] += 1.0
            self.observed[y, x] = True
        np.clip(self.log_odds, -4, 4, out=self.log_odds)
        if np.any(before != self.occupied):
            self.revision += 1

    def clearance(self):
        obstacles = self.occupied.copy()
        obstacles[[0, -1], :] = True
        obstacles[:, [0, -1]] = True
        return distance_transform_edt(~obstacles) * self.config.resolution

    def traversable(self, robot=ROBOT):
        return self.clearance() > robot.footprint_radius + robot.safety_margin
