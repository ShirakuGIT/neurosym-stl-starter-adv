import torch
import argparse
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm
import numpy as np

from scenarios import generate_env, generate_hard_env_and_spec
from rrt_bootstrap import find_knots_rrt

def generate_one_trajectory(args_tuple):
    """Worker function for multiprocessing."""
    i, seed, args = args_tuple
    torch.manual_seed(seed + i)
    np.random.seed(seed + i)

    env = None
    attempts = 0
    max_attempts = 10
    
    while env is None and attempts < max_attempts:
        try:
            if args.use_hard_cases:
                formula_types = ['default', 'sequence', 'until']
                formula = formula_types[i % len(formula_types)]
                env = generate_hard_env_and_spec(T=args.T, K=args.K, seed=seed+i, formula_type=formula)
            else:
                env = generate_env(args.num_obstacles, seed=seed+i, T=args.T, K=args.K)
        except Exception as e:
            attempts += 1
            if attempts >= max_attempts:
                return None
    
    try:
        knots = find_knots_rrt(env, args.rrt_time, args.rrt_iter)
        if knots is None:
            return None
        return (env.spec.get_cond_vec(), knots)
    except Exception:
        return None

def main(args):
    """Main function to generate the dataset."""
    dataset = []
    
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    pool = Pool(processes=args.num_workers)
    tasks = [(i, np.random.randint(1e6), args) for i in range(args.num)]
    
    with tqdm(total=args.num, desc="Generating Trajectories") as pbar:
        for result in pool.imap_unordered(generate_one_trajectory, tasks):
            if result is not None:
                dataset.append(result)
            pbar.update(1)

    pool.close()
    pool.join()

    print(f"\nGenerated {len(dataset)} valid trajectories out of {args.num} attempts.")
    
    if len(dataset) > 0:
        torch.save(dataset, args.out)
        print(f"Dataset saved to {args.out}")
    else:
        print("No trajectories were generated. Check RRT parameters and environment complexity.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='./datasets/expert_knots.pth')
    parser.add_argument('--num', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    parser.add_argument('--use_hard_cases', action='store_true', help='Use the hard case scenario generator')
    
    # Env params
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--T', type=int, default=64)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--num_obstacles', type=int, default=5)
    
    # RRT params
    parser.add_argument('--rrt_time', type=float, default=0.5)
    parser.add_argument('--rrt_iter', type=int, default=2000)
    
    args = parser.parse_args()
    main(args)