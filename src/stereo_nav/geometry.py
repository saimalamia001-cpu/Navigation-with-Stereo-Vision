import math
import numpy as np


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def integrate(pose, v, w, dt):
    """Exact unicycle integration, including the straight-line limit."""
    x, y, yaw = pose
    if abs(w) < 1e-8:
        return np.array([x + v * dt * math.cos(yaw), y + v * dt * math.sin(yaw), yaw])
    return np.array([x + v / w * (math.sin(yaw + w * dt) - math.sin(yaw)),
                     y - v / w * (math.cos(yaw + w * dt) - math.cos(yaw)),
                     wrap(yaw + w * dt)])


def wheel_rates(v, w, radius, track):
    return (v - w * track / 2) / radius, (v + w * track / 2) / radius


def encoder_update(pose, dl, dr, radius, track):
    left, right = dl * radius, dr * radius
    return integrate(pose, (left + right) / 2, (right - left) / track, 1.0)


def ray_cells(a, b):
    """Integer Bresenham traversal; includes origin and endpoint."""
    x, y = map(int, a)
    x1, y1 = map(int, b)
    dx, dy = abs(x1 - x), -abs(y1 - y)
    sx, sy = (1 if x < x1 else -1), (1 if y < y1 else -1)
    err = dx + dy
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
