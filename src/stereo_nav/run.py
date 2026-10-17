import argparse
import csv
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .config import CAMERA
from .mapping import OccupancyGrid
from .perception import StereoDepth
from .planning import DynamicWindow, astar
from .render import dashboard
from .simulation import Simulation
from .evaluation import map_metrics


def run(output, scenario='slalom', seed=7, max_seconds=100, gui=False, video=True, frames_every=2,
        replan_policy='online', use_imu=True):
    output = Path(output)
    if max_seconds <= 0 or frames_every < 1:
        raise ValueError('Duration must be positive and frames_every must be at least one')
    output.mkdir(parents=True, exist_ok=True)
    sim = Simulation(output, seed, scenario, gui, use_imu=use_imu)
    mapper, stereo, controller = OccupancyGrid(), StereoDepth(), DynamicWindow()
    pose = sim.odometry.copy()
    path = np.empty((0, 2))
    writer = None
    rows, samples = [], []
    totals = dict(depth_count=0, depth_roi_count=0, depth_abs_sum=0., depth_squared_sum=0.)
    replans, last_revision = 0, -1
    status = 'timeout'
    started = time.perf_counter()
    try:
        if video:
            writer = cv2.VideoWriter(str(output / 'replay.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                     5 / frames_every, (1280, 1000))
            if not writer.isOpened():
                raise RuntimeError('Could not open video encoder')
        for step in range(max(1, round(max_seconds / .2))):
            tick = time.perf_counter()
            left, right = sim.capture_rgb()
            obs = stereo.compute(left, right)
            perception_ms = (time.perf_counter() - tick) * 1000
            mapper.update(pose, obs.points)
            # Replan when new evidence invalidates the remaining route. Keeping
            # a valid route avoids oscillating between equally good detours as
            # observations elsewhere change the map's unknown-space costs.
            invalid_path = not len(path)
            if len(path) and mapper.revision != last_revision:
                nearest = np.argmin(np.linalg.norm(path - pose[:2], axis=1))
                cells = mapper.cell(path[nearest:])
                free = mapper.traversable()
                invalid_path = not free[cells[:, 1], cells[:, 0]].all()
            should_plan = replans == 0 or (replan_policy == 'online' and invalid_path)
            if should_plan:
                path = astar(mapper, pose[:2], sim.goal)
                replans += 1
            last_revision = mapper.revision
            cmd = controller.command(pose, path, mapper)
            # Explicit blindness gate. Invalid disparity is never interpreted as free.
            yy, xx = np.indices(obs.valid.shape)
            central = (abs(xx - CAMERA.width / 2) < 70) & (yy > 65) & (yy < 155)
            front_valid = (obs.valid & central).sum() / central.sum()
            if front_valid < .015:
                cmd = (0., cmd[1])
                controller.previous[0] = 0
            for key, value in sim.evaluate_depth(obs).items():
                totals[key] += value
            loop_ms = (time.perf_counter() - tick) * 1000
            samples.append(loop_ms)
            row = dict(step=step, time_s=round(sim.time, 3), x=float(pose[0]), y=float(pose[1]), yaw=float(pose[2]),
                       v=cmd[0], w=cmd[1], valid_fraction=float(obs.valid.mean()),
                       perception_ms=perception_ms, loop_ms=loop_ms, map_revision=mapper.revision,
                       path_points=len(path))
            rows.append(row)
            if step % frames_every == 0:
                frame = dashboard(left, right, obs, mapper, pose, path, sim, cmd, controller.reason)
                if writer:
                    writer.write(frame)
                cv2.imwrite(str(output / 'latest.jpg'), frame)
            if step % 50 == 0:
                print(f'{scenario} seed={seed} t={sim.time:.1f}s pose={pose[:2].round(2)} path={len(path)} v={cmd[0]:.2f} w={cmd[1]:.2f}', flush=True)
            if np.linalg.norm(pose[:2] - sim.goal) < .3:
                # Evaluate success against simulator truth, not odometry alone.
                status = 'success' if np.linalg.norm(sim.true_pose()[:2] - sim.goal) < .45 else 'odometry_failure'
                break
            pose = sim.step(*cmd)
            if sim.collided:
                status = 'collision'
                break
        n = max(totals['depth_count'], 1)
        metrics = dict(scenario=scenario, seed=seed, scene_style=sim.scene_style,
                       replan_policy=replan_policy, imu_fusion=use_imu,
                       status=status, collision=sim.collided,
                       simulated_seconds=sim.time, wall_seconds=time.perf_counter() - started,
                       distance_m=float(sim.distance), minimum_clearance_m=float(sim.closest),
                       final_goal_error_m=float(np.linalg.norm(sim.true_pose()[:2] - sim.goal)),
                       odometry_error_m=float(np.linalg.norm(pose[:2] - sim.true_pose()[:2])),
                       replans=replans, map_revisions=mapper.revision,
                       depth_mae_m=totals['depth_abs_sum'] / n if totals['depth_count'] else None,
                       depth_rmse_m=(totals['depth_squared_sum'] / n) ** .5 if totals['depth_count'] else None,
                       obstacle_depth_coverage=totals['depth_count'] / max(totals['depth_roi_count'], 1),
                       loop_median_ms=float(np.median(samples)), loop_p95_ms=float(np.percentile(samples, 95)),
                       depth_evaluated_pixels=totals['depth_count'],
                       depth_eligible_pixels=totals['depth_roi_count'])
        metrics.update(map_metrics(mapper, sim.boxes))
        # Hold a terminal frame so the replay and preview clearly show outcome.
        final_frame = dashboard(left, right, obs, mapper, pose, path, sim, (0., 0.), status)
        cv2.imwrite(str(output / 'latest.jpg'), final_frame)
        if writer:
            for _ in range(max(1, round(5 / frames_every))):
                writer.write(final_frame)
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        with (output / 'telemetry.csv').open('w', newline='', encoding='utf-8') as f:
            csv_writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            csv_writer.writeheader()
            csv_writer.writerows(rows)
        np.savez_compressed(output / 'map.npz', log_odds=mapper.log_odds, observed=mapper.observed,
                            trajectory=np.asarray(sim.truth_path), path=path)
        cv2.imwrite(str(output / 'left.png'), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output / 'right.png'), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output / 'overview.png'), cv2.cvtColor(sim.overview_rgb(), cv2.COLOR_RGB2BGR))
        np.save(output / 'depth.npy', obs.depth)
        print(json.dumps(metrics, indent=2), flush=True)
        return metrics
    finally:
        if writer:
            writer.release()
        sim.close()


def main():
    p = argparse.ArgumentParser(description='Stereo TurtleBot-style navigation in PyBullet')
    p.add_argument('--scenario', choices=['slalom', 'corridor', 'blocked', 'empty', 'reveal'], default='slalom')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--seconds', type=float, default=100)
    p.add_argument('--output', type=Path, default=Path('runs/demo'))
    p.add_argument('--gui', action='store_true')
    p.add_argument('--no-video', action='store_true')
    p.add_argument('--replan-policy', choices=['online', 'fixed'], default='online',
                   help='fixed freezes the first global path; local obstacle avoidance remains active')
    p.add_argument('--wheel-only', action='store_true', help='Disable gyro fusion for an odometry ablation')
    args = p.parse_args()
    if args.seconds <= 0:
        p.error('--seconds must be positive')
    run(args.output, args.scenario, args.seed, args.seconds, args.gui, not args.no_video,
        replan_policy=args.replan_policy, use_imu=not args.wheel_only)


if __name__ == '__main__':
    main()
