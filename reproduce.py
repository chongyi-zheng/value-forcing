"""Generate commands or Slurm launchers for the paper's experiments.

By default, this script prints one main.py command for each experiment. Add --slurm to create a submission
script for each method. Run a generated script with bash to submit its experiments to Slurm, with one experiment
per array task. Examples:

    python reproduce.py ogbench --data_root=~/data/ogbench > ogbench.txt
    python reproduce.py ogbench --data_root=~/data/ogbench --methods vf ivf --slurm
    bash sbatch/vf_ogbench.sh
    python reproduce.py ocbench --data_root=~/data/ocbench --methods ivf --slurm --partition=gpu

Extra training flags, such as --save_interval=50000, are passed directly to every main.py command.
"""

import argparse
import os
import shlex

from utils.slurm import write_slurm_launcher

# Each domain lists its environment name prefix, dataset folder, and training updates between shard switches.
# All domains have tasks 1 through 5 and 100M transitions. Most datasets have 100 shards of 1M transitions;
# humanoidmaze-giant has 25 shards of 4M. The default 500,000 updates cycle through the full shard list five times.
OGBENCH_DOMAINS = {
    'cube-triple': ('cube-triple-play', 'cube-triple-play-100m-v0', 1000),
    'cube-quadruple': ('cube-quadruple-play', 'cube-quadruple-play-100m-v0', 1000),
    'puzzle-4x4': ('puzzle-4x4-play', 'puzzle-4x4-play-100m-v0', 1000),
    'humanoidmaze-giant': ('humanoidmaze-giant-navigate', 'humanoidmaze-giant-navigate-100m-v0', 4000),
}

# Each method lists its agent file and any settings that change for specific domains.
# The agent files already use the cube-triple settings, so only differences from those defaults appear here.
OGBENCH_METHODS = {
    'ivf': (
        'agents/implicit_value_forcing.py',
        {
            'cube-quadruple': {'expectile': 0.8},
            'humanoidmaze-giant': {'expectile': 0.8},
        },
    ),
    'vf': ('agents/value_forcing.py', {}),
    'floq': (
        'agents/floq.py',
        {
            'cube-quadruple': {'clip_return_embed': True},
            'puzzle-4x4': {'alpha': 3000.0, 'noise_coverage': 0.25, 'block_depth': 4},
            'humanoidmaze-giant': {'block_depth': 4},
        },
    ),
    'flow_critic': ('agents/flow_critic.py', {}),
    'value_flows': (
        'agents/value_flows.py',
        {
            'puzzle-4x4': {'bcfm_lambda': 3.0, 'confidence_weight_temp': 1.0},
            'humanoidmaze-giant': {'bcfm_lambda': 3.0},
        },
    ),
    'pcbf': (
        'agents/pcbf.py',
        {
            'puzzle-4x4': {'lambda_param': 0.4},
            'humanoidmaze-giant': {'lambda_param': 0.2},
        },
    ),
    'c51': ('agents/c51.py', {}),
    'iqn': ('agents/iqn.py', {'humanoidmaze-giant': {'kappa': 30.0}}),
    'qc': ('agents/qc.py', {}),
    'divl': ('agents/divl.py', {}),
    'implicit_c51': (
        'agents/implicit_c51.py',
        {
            'cube-quadruple': {'expectile': 0.8},
            'humanoidmaze-giant': {'expectile': 0.8},
        },
    ),
    'implicit_iqn': ('agents/implicit_iqn.py', {'humanoidmaze-giant': {'kappa': 30.0}}),
    'implicit_qc': ('agents/implicit_qc.py', {}),
}

# The six OCBench lite tasks used in the experiments.
OCBENCH_TASKS = [
    'block-lite-single-task1-v0',
    'block-lite-double-task1-v0',
    'block-lite-double-task2-v0',
    'chamber-lite-easy-task1-v0',
    'chamber-lite-easy-task2-v0',
    'chamber-lite-easy-task3-v0',
]
OCBENCH_METHODS = ['vf', 'value_flows', 'c51', 'qc', 'ivf', 'implicit_c51', 'implicit_qc']
# Settings shared by all OCBench methods; other agent settings keep their defaults.
# The residual networks use four blocks of two layers, with 1,024 units per layer.
OCBENCH_AGENT_FLAGS = {
    'discount': 0.9999,
    'mlp_class': 'resmlp',
    'actor_hidden_dims': (1024,) * 8,
    'value_hidden_dims': (1024,) * 8,
}
# MJWarp runs the 500 evaluation episodes in parallel at each evaluation.
OCBENCH_MAIN_FLAGS = {'dataset_replace_interval': 10000, 'eval_interval': 50000, 'eval_episodes': 500}

# Convert each data budget from a percentage to the fraction passed to --dataset_subset_fraction.
# Keep whole trajectories from each shard, up to the requested budget. For 100%, omit the flag and use all data.
BUDGETS = {'1': 0.01, '5': 0.05, '10': 0.1, '25': 0.25, '50': 0.5, '100': None}
DEFAULT_BUDGETS = {'ogbench': ['10', '25', '50', '100'], 'ocbench': ['25', '100']}
SEEDS = [100, 200, 300, 400]


def flag_value(value):
    """Format a flag value, including tuples such as network layer sizes."""
    if isinstance(value, tuple):
        return '(' + ', '.join(str(v) for v in value) + ')'
    return str(value)


def command(
    agent_file,
    env_name,
    dataset_dir,
    seed,
    budget,
    save_dir,
    group,
    main_flags,
    agent_flags,
    extra,
    *,
    python_executable='python',
):
    """Build a command for one experiment, with paths and argument values quoted for the shell."""
    # EGL lets MuJoCo render on machines without a display.
    args = ['env', 'MUJOCO_GL=egl', python_executable, 'main.py', f'--agent={agent_file}', f'--env_name={env_name}']
    args.append(f'--dataset_dir={dataset_dir}')
    args += [f'--{k}={flag_value(v)}' for k, v in main_flags.items()]
    args += [f'--seed={seed}', f'--save_dir={save_dir}', f'--wandb_run_group={group}']
    if BUDGETS[budget] is not None:
        args.append(f'--dataset_subset_fraction={BUDGETS[budget]}')
    args += [f'--agent.{k}={flag_value(v)}' for k, v in agent_flags.items()]
    # Put extra flags last so they can override the generated settings.
    return shlex.join(args + extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0], allow_abbrev=False)
    parser.add_argument('suite', choices=['ogbench', 'ocbench'])
    parser.add_argument('--data_root', required=True, help='Directory holding the datasets (see the README).')
    parser.add_argument('--save_root', default='exp', help='Root of the run directories.')
    parser.add_argument('--methods', nargs='+', help='Methods to run (default: all of the suite).')
    parser.add_argument('--domains', nargs='+', choices=list(OGBENCH_DOMAINS), help='OGBench domains (default: all).')
    parser.add_argument('--tasks', nargs='+', help='Task numbers (OGBench) or task names (OCBench); default: all.')
    parser.add_argument('--budgets', nargs='+', choices=list(BUDGETS), help='Data budgets in percent.')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--slurm', action='store_true', help='Write submission scripts instead of printing commands.')
    parser.add_argument('--output_dir', default='sbatch', help='Directory for the generated Slurm scripts and logs.')
    parser.add_argument('--python', dest='python_executable', help='Path to Python (Slurm default: .venv/bin/python).')
    parser.add_argument('--partition', help='Slurm partition (default: the cluster default).')
    parser.add_argument('--account', help='Slurm account (default: the cluster default).')
    parser.add_argument('--gres', default='gpu:1', help='Slurm GPU request, such as gpu:a100:1.')
    parser.add_argument('--cpus_per_task', type=int, default=4, help='CPUs per experiment.')
    parser.add_argument('--mem', default='64G', help='Memory per experiment.')
    parser.add_argument('--time', default='24:00:00', help='Time limit per experiment.')
    parser.add_argument(
        '--max_parallel', type=int, help='Maximum simultaneous experiments per array (default: no limit).'
    )
    args, extra = parser.parse_known_args()

    methods = OGBENCH_METHODS if args.suite == 'ogbench' else OCBENCH_METHODS
    for method in args.methods or []:
        if method not in methods:
            parser.error(f'unknown {args.suite} method {method!r}; choose from {list(methods)}')
    budgets = args.budgets or DEFAULT_BUDGETS[args.suite]
    data_root = os.path.expanduser(args.data_root)
    save_root = os.path.expanduser(args.save_root)
    repo_root = os.path.dirname(os.path.realpath(__file__))
    python_executable = args.python_executable or 'python'
    if args.slurm:
        if args.cpus_per_task <= 0 or (args.max_parallel is not None and args.max_parallel <= 0):
            parser.error('--cpus_per_task and --max_parallel must be positive when specified')
        # Keep these paths valid when Slurm runs the command from the repository directory.
        data_root = os.path.abspath(data_root)
        save_root = os.path.abspath(save_root)
        # Keep the Python path inside the virtual environment; following its symlink can bypass the environment.
        python_executable = os.path.abspath(
            os.path.expanduser(args.python_executable or os.path.join(repo_root, '.venv', 'bin', 'python'))
        )
    commands = {method: [] for method in args.methods or methods}

    if args.suite == 'ogbench':
        for method in args.methods or list(OGBENCH_METHODS):
            agent_file, domain_flags = OGBENCH_METHODS[method]
            for domain in args.domains or list(OGBENCH_DOMAINS):
                prefix, dataset, interval = OGBENCH_DOMAINS[domain]
                for task in args.tasks or ['1', '2', '3', '4', '5']:
                    env_name = f'{prefix}-singletask-task{task}-v0'
                    for budget in budgets:
                        save_dir = os.path.join(save_root, 'ogbench', method, env_name, f'budget{budget}')
                        for seed in args.seeds:
                            commands[method].append(
                                command(
                                    agent_file,
                                    env_name,
                                    os.path.join(data_root, dataset),
                                    seed,
                                    budget,
                                    save_dir,
                                    f'ogbench_{method}',
                                    {'dataset_replace_interval': interval},
                                    domain_flags.get(domain, {}),
                                    extra,
                                    python_executable=python_executable,
                                )
                            )
    else:
        for method in args.methods or OCBENCH_METHODS:
            agent_file = OGBENCH_METHODS[method][0]
            for env_name in args.tasks or OCBENCH_TASKS:
                for budget in budgets:
                    save_dir = os.path.join(save_root, 'ocbench', method, env_name, f'budget{budget}')
                    for seed in args.seeds:
                        commands[method].append(
                            command(
                                agent_file,
                                env_name,
                                os.path.join(data_root, 'lite', env_name),
                                seed,
                                budget,
                                save_dir,
                                f'ocbench_{method}',
                                OCBENCH_MAIN_FLAGS,
                                OCBENCH_AGENT_FLAGS,
                                extra,
                                python_executable=python_executable,
                            )
                        )

    for method, method_commands in commands.items():
        if args.slurm:
            path = write_slurm_launcher(
                method_commands,
                os.path.join(args.output_dir, f'{method}_{args.suite}.sh'),
                repo_root=repo_root,
                job_name=f'{method}_{args.suite}',
                partition=args.partition,
                account=args.account,
                gres=args.gres,
                cpus_per_task=args.cpus_per_task,
                mem=args.mem,
                time_limit=args.time,
                max_parallel=args.max_parallel,
            )
            print(f'Wrote {len(method_commands)} experiments to {path}')
            print(f'Submit with: bash {shlex.quote(str(path))}')
        else:
            print('\n'.join(method_commands))


if __name__ == '__main__':
    main()
