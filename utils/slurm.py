"""Create self-contained Slurm submission scripts for experiment commands."""

import os
import shlex


def write_slurm_launcher(
    commands,
    output_path,
    *,
    repo_root,
    job_name,
    partition=None,
    account=None,
    gres='gpu:1',
    cpus_per_task=4,
    mem='64G',
    time_limit='24:00:00',
    max_parallel=None,
):
    """Write a launcher that submits one experiment per array task when run with bash."""
    if isinstance(commands, str):
        raise ValueError('commands must be a sequence of shell commands')
    commands = list(commands)
    if not commands or any(not isinstance(command, str) or not command.strip() for command in commands):
        raise ValueError('commands must contain at least one nonempty shell command')
    if any('\n' in command or '\r' in command for command in commands):
        raise ValueError('each experiment command must fit on one line')
    for name, value in [('cpus_per_task', cpus_per_task), ('max_parallel', max_parallel)]:
        if name == 'max_parallel' and value is None:
            continue
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    for name, value in [
        ('job_name', job_name),
        ('partition', partition),
        ('account', account),
        ('gres', gres),
        ('mem', mem),
        ('time_limit', time_limit),
    ]:
        if value is not None and (not isinstance(value, str) or '\n' in value or '\r' in value):
            raise ValueError(f'{name} must be a single-line string')

    repo_root = os.path.realpath(os.path.expanduser(repo_root))
    output_path = os.path.realpath(os.path.expanduser(output_path))
    if any('\n' in path or '\r' in path for path in (repo_root, output_path)):
        raise ValueError('repository and output paths must fit on one line')
    output_dir = os.path.dirname(output_path)
    log_dir = os.path.join(output_dir, 'logs')
    array = f'0-{len(commands) - 1}'
    if max_parallel is not None:
        array += f'%{max_parallel}'
    submit = [
        'sbatch',
        f'--job-name={job_name}',
        f'--array={array}',
        '--ntasks=1',
        f'--gres={gres}',
        f'--cpus-per-task={cpus_per_task}',
        f'--mem={mem}',
        f'--time={time_limit}',
        f'--output={log_dir}/%x-%A_%a.out',
        f'--error={log_dir}/%x-%A_%a.err',
    ]
    if partition:
        submit.append(f'--partition={partition}')
    if account:
        submit.append(f'--account={account}')

    lines = [
        '#!/usr/bin/env bash',
        'set -euo pipefail',
        '',
        f'mkdir -p -- {shlex.quote(log_dir)}',
        f'exec {shlex.join(submit)} "$@" <<\'VALUE_FORCING_SLURM_JOB\'',
        '#!/usr/bin/env bash',
        'set -euo pipefail',
        '',
        f'cd -- {shlex.quote(repo_root)}',
        'export OMP_NUM_THREADS=1',
        'export OPENBLAS_NUM_THREADS=1',
        'export XLA_PYTHON_CLIENT_PREALLOCATE=false',
        'export XLA_CLIENT_MEM_FRACTION=0.8',
        '',
        'commands=(',
        *[f'    {shlex.quote(command)}' for command in commands],
        ')',
        'task_id=${SLURM_ARRAY_TASK_ID:-}',
        'last_task_id=$((${#commands[@]} - 1))',
        'if [[ ! "$task_id" =~ ^(0|[1-9][0-9]*)$ ]] ||',
        '   (( ${#task_id} > ${#last_task_id} )) || (( task_id > last_task_id )); then',
        '    printf \'Invalid SLURM_ARRAY_TASK_ID: %s (expected 0 through %s)\\n\' "$task_id" "$last_task_id" >&2',
        '    exit 1',
        'fi',
        'exec srun --ntasks=1 bash -c "${commands[$task_id]}"',
        'VALUE_FORCING_SLURM_JOB',
        '',
    ]
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as script:
        script.write('\n'.join(lines))
    os.chmod(output_path, os.stat(output_path).st_mode | 0o111)
    return output_path
