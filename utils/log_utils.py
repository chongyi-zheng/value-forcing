import csv
import json
import os
import shutil
import tempfile
from datetime import datetime

import absl.flags as flags
import ml_collections
import numpy as np
import wandb
from PIL import Image, ImageEnhance


class CsvLogger:
    """Write metrics to CSV, optionally continuing from a saved training step."""

    def __init__(self, path, resume_step=None):
        """Keep existing rows through `resume_step` when continuing a run."""
        self.path = path
        self.header = None
        self.file = None
        self.writer = None
        self.last_row = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)
        if resume_step is not None:
            self._prepare_resume(resume_step)

    def _prepare_resume(self, resume_step):
        """Validate prior metrics and back up any rows removed for a restart."""
        if not os.path.exists(self.path) or os.path.getsize(self.path) == 0:
            return
        with open(self.path, newline='') as file:
            lines = file.readlines()
        reader = csv.reader(lines, strict=True)
        try:
            header = next(reader)
        except (StopIteration, csv.Error) as error:
            raise ValueError(f'Cannot resume malformed CSV header: {self.path}') from error
        if not header or 'step' not in header or len(set(header)) != len(header) or any(not key for key in header):
            raise ValueError(f'Cannot resume CSV without a unique step column and valid header: {self.path}')

        step_column = header.index('step')
        rows = []
        unterminated_last_line = not lines[-1].endswith(('\n', '\r'))
        needs_repair = unterminated_last_line
        while True:
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error as error:
                # A crash can leave the final quoted field unfinished.
                if str(error) == 'unexpected end of data' and reader.line_num == len(lines):
                    needs_repair = True
                    break
                raise ValueError(f'Cannot resume malformed CSV: {self.path}') from error
            # Even a parseable final record may have a partially written step.
            if unterminated_last_line and reader.line_num == len(lines):
                break
            try:
                if len(row) != len(header):
                    raise ValueError('wrong number of columns')
                step = int(row[step_column])
            except ValueError as error:
                if reader.line_num == len(lines):
                    needs_repair = True
                    break
                raise ValueError(
                    f'Cannot resume malformed CSV row ending at line {reader.line_num}: {self.path}'
                ) from error
            if step <= resume_step:
                rows.append(row)
            else:
                needs_repair = True

        if needs_repair:
            directory = os.path.dirname(os.path.abspath(self.path))
            basename = os.path.basename(self.path)
            backup_fd, backup_path = tempfile.mkstemp(prefix=f'{basename}.backup-', dir=directory)
            os.close(backup_fd)
            shutil.copy2(self.path, backup_path)
            temporary_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode='w', newline='', prefix=f'.{basename}.', dir=directory, delete=False
                ) as file:
                    temporary_path = file.name
                    writer = csv.writer(file, lineterminator='\n')
                    writer.writerow(header)
                    writer.writerows(rows)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temporary_path, self.path)
            finally:
                if temporary_path is not None and os.path.exists(temporary_path):
                    os.unlink(temporary_path)
        self.header = header
        self.last_row = dict(zip(header, rows[-1])) if rows else None

    def log(self, row, step):
        """Add the step and append values in the original column order."""
        row['step'] = step
        if self.file is None:
            if self.header is None:
                self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
                self.file = open(self.path, 'w', newline='')
                self.writer = csv.writer(self.file, lineterminator='\n')
                self.writer.writerow(self.header)
            else:
                self.file = open(self.path, 'a', newline='')
                self.writer = csv.writer(self.file, lineterminator='\n')
        filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
        values = [str(filtered_row.get(k, '')) for k in self.header]
        self.writer.writerow(values)
        self.file.flush()
        self.last_row = dict(zip(self.header, values))

    def close(self):
        """Close the CSV file if it has been opened."""
        if self.file is not None:
            self.file.close()


def get_exp_name(seed):
    """Build an experiment name from the seed, optional SLURM IDs, and current time."""
    exp_name = ''
    exp_name += f'sd{seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    return exp_name


def get_flag_dict():
    """Collect top-level flags, converting ConfigDict values to dictionaries."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_wandb(
    wandb_output_dir=None,
    entity=None,
    project='project',
    group=None,
    name=None,
    mode='online',
    resume=False,
):
    """Start W&B or reconnect to the run saved in the experiment directory."""
    if wandb_output_dir is None:
        wandb_output_dir = tempfile.mkdtemp()
    identity_path = os.path.join(wandb_output_dir, 'wandb_run.json')
    identity = None
    online = mode not in ('offline', 'dryrun', 'disabled')
    if resume and mode != 'disabled':
        with open(identity_path) as file:
            identity = json.load(file)
        if not isinstance(identity, dict) or not identity.get('id') or not identity.get('project'):
            raise ValueError(f'Invalid W&B run identity in {identity_path}.')

    tags = [group] if group is not None else None

    init_kwargs = dict(
        config=get_flag_dict(),
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_stats=False,
        ),
        mode=mode,
        save_code=True,
    )

    if identity is not None:
        init_kwargs.update(id=identity['id'], entity=identity.get('entity') or entity, project=identity['project'])
        if online:
            init_kwargs['resume'] = 'must' if identity.get('online') else 'allow'
        else:
            print('[wandb] offline mode keeps the run ID but cannot resume online history.')

    run = wandb.init(**init_kwargs)
    if mode == 'disabled':
        return run

    identity = dict(
        id=run.id,
        entity=run.entity or (identity or {}).get('entity'),
        project=run.project,
        online=online or bool((identity or {}).get('online')),
    )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=wandb_output_dir, prefix='.wandb_run.', delete=False) as file:
            temporary_path = file.name
            json.dump(identity, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, identity_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return run


def reshape_video(v, n_cols=None):
    """Tile videos into a grid with shape (T, C, H_grid, W_grid).

    Accept a single (T, H, W, C) video or a batch with shape (N, T, H, W, C).
    """
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, (n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, (t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    """Pad and tile rendered videos into a W&B video, updating `renders` in place."""
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Dim padding frames to mark completed episodes.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    return wandb.Video(renders, fps=fps, format='mp4')
