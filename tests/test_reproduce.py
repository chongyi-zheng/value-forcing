"""Regression checks for generated reproduction commands; no jobs are submitted."""

import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReproduceTest(unittest.TestCase):
    def test_home_directory_save_root_in_printed_and_slurm_commands(self):
        expected_save_dir = os.path.join(
            os.path.expanduser('~/value-forcing-test-output'),
            'ogbench',
            'vf',
            'cube-triple-play-singletask-task1-v0',
            'budget25',
        )
        with tempfile.TemporaryDirectory() as output_dir:
            for slurm in (False, True):
                with self.subTest(slurm=slurm):
                    args = [
                        sys.executable,
                        str(REPO_ROOT / 'reproduce.py'),
                        'ogbench',
                        '--data_root=/unused/data',
                        '--save_root=~/value-forcing-test-output',
                        '--methods',
                        'vf',
                        '--domains',
                        'cube-triple',
                        '--tasks',
                        '1',
                        '--budgets',
                        '25',
                        '--seeds',
                        '100',
                    ]
                    if slurm:
                        args += ['--slurm', f'--output_dir={output_dir}']
                    result = subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=True)
                    if slurm:
                        script_path = Path(output_dir) / 'vf_ogbench.sh'
                        subprocess.run(['bash', '-n', str(script_path)], check=True, capture_output=True)
                        script = script_path.read_text()
                        array = script.split('commands=(\n', 1)[1].split('\n)', 1)[0]
                        commands = shlex.split(array)
                    else:
                        commands = result.stdout.splitlines()
                    self.assertEqual(len(commands), 1)
                    argv = shlex.split(commands[0])
                    self.assertIn(f'--save_dir={expected_save_dir}', argv)


if __name__ == '__main__':
    unittest.main()
