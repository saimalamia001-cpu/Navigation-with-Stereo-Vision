"""Reproducible multi-scenario evaluation; failures stay in the report."""
import argparse
import json
from pathlib import Path
import numpy as np
from .run import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=[7, 19, 42])
    p.add_argument('--scenarios', nargs='+', default=['empty', 'slalom', 'corridor', 'blocked', 'reveal'])
    p.add_argument('--seconds', type=float, default=100)
    p.add_argument('--output', type=Path, default=Path('runs/benchmark'))
    p.add_argument('--compare-fixed', action='store_true')
    p.add_argument('--wheel-only', action='store_true')
    args = p.parse_args()
    results = []
    policies = ['online', 'fixed'] if args.compare_fixed else ['online']
    for policy in policies:
        for scenario in args.scenarios:
            for seed in args.seeds:
                output = args.output / f'{scenario}-{seed}-{policy}'
                results.append(run(output, scenario, seed, args.seconds, video=False, frames_every=10,
                                   replan_policy=policy, use_imu=not args.wheel_only))
                (args.output / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    summary = {}
    for policy in policies:
        subset = [r for r in results if r['replan_policy'] == policy]
        success = [r for r in subset if r['status'] == 'success']
        depth_measured = [r['depth_mae_m'] for r in subset if r['depth_mae_m'] is not None]
        summary[policy] = dict(runs=len(subset), successes=len(success),
                               success_rate=len(success) / len(subset),
                               collisions=sum(r['collision'] for r in subset),
                               successful_mean_distance_m=float(np.mean([r['distance_m'] for r in success])) if success else None,
                               mean_depth_mae_m=float(np.mean(depth_measured)) if depth_measured else None,
                               depth_error_runs=len(depth_measured),
                               failures=[{'scenario': r['scenario'], 'seed': r['seed'], 'status': r['status']}
                                         for r in subset if r['status'] != 'success'])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
