"""Offline system identification; navigation itself never consumes true pose."""
import json
from pathlib import Path
import numpy as np
from .geometry import wrap
from .simulation import Simulation


def trial(output, v, w, seconds=8):
    sim = Simulation(output, seed=123, scenario='empty', use_imu=False)
    try:
        enc = sim.encoders.copy()
        previous = sim.true_pose()
        distance, yaw = 0., 0.
        for _ in range(round(seconds / .2)):
            sim.step(v, w)
            current = sim.true_pose()
            distance += np.linalg.norm(current[:2] - previous[:2])
            yaw += wrap(current[2] - previous[2])
            previous = current
        delta = sim.encoders - enc
        return dict(left_rad=float(delta[0]), right_rad=float(delta[1]),
                    distance_m=float(distance), yaw_rad=float(yaw),
                    odometry_error_m=float(np.linalg.norm(sim.odometry[:2] - current[:2])))
    finally:
        sim.close()


def main():
    output = Path('runs/calibration')
    straight = trial(output / 'straight', .3, 0)
    spin = trial(output / 'spin', 0, .6)
    radius = straight['distance_m'] / ((straight['left_rad'] + straight['right_rad']) / 2)
    track = radius * (spin['right_rad'] - spin['left_rad']) / spin['yaw_rad']
    # Curved motion is held out of the parameter fit.
    curved = trial(output / 'held-out-arc', .3, .5)
    report = dict(straight=straight, spin=spin, held_out_arc=curved,
                  identified_radius_m=radius, identified_track_m=track,
                  note='Uses simulator truth OFFLINE for system identification only. Parameters are fixed during navigation.')
    (output / 'calibration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
