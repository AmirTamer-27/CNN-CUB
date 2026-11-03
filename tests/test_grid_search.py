"""Exercise process scheduling without CUDA, datasets, or MLflow access."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

from src.training.grid_search import run_grid_search


class GridSearchTests(unittest.TestCase):
    def test_four_trials_reuse_two_gpus_and_preserve_settings(self):
        launched = []
        active = set()

        class Process:
            def __init__(self, command, env, **kwargs):
                self.gpu = env['CUDA_VISIBLE_DEVICES']
                assert self.gpu not in active
                active.add(self.gpu)
                trial = yaml.safe_load(Path(command[-1]).read_text())
                launched.append((self.gpu, trial, env))
                self.polls = 0

            def poll(self):
                return None

        # Scheduling requires two polls before a GPU becomes free.
        def poll(process):
            process.polls += 1
            if process.polls == 1:
                return None
            active.discard(process.gpu)
            return 0

        config = dict(model='bird_cnn_delayed_pooling', batch_size=32, seed=42,
                      weight_decay=0.0001, lr_scheduler='plateau')
        original = config.copy()
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch('torch.cuda.is_available', return_value=True), \
                     patch('torch.cuda.device_count', return_value=2), \
                     patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-A,GPU-B', 'MLFLOW_RUN_ID': 'parent'}), \
                     patch('src.training.grid_search.subprocess.Popen', Process), \
                     patch.object(Process, 'poll', poll), \
                     patch('src.training.grid_search.time.sleep'):
                    run_grid_search(config, [0, 1], 120, [0.0005, 0.001], [0.05, 0.1])
                self.assertEqual(len(list(Path('outputs/grid_search').glob('*/*.yaml'))), 4)
                self.assertEqual(len(list(Path('.').rglob('*.json'))), 0)
            finally:
                os.chdir(cwd)
        self.assertEqual(config, original)
        self.assertEqual([gpu for gpu, _, _ in launched], ['GPU-A', 'GPU-B'] * 2)
        self.assertEqual({(trial['learning_rate'], trial['dropout2d']) for _, trial, _ in launched},
                         {(0.0005, 0.05), (0.0005, 0.1), (0.001, 0.05), (0.001, 0.1)})
        self.assertEqual(len({trial['run_name'] for _, trial, _ in launched}), 4)
        for _, trial, env in launched:
            self.assertEqual(trial['epochs'], 120)
            self.assertEqual(trial['batch_size'], 32)
            self.assertEqual(trial['weight_decay'], 0.0001)
            self.assertNotIn('MLFLOW_RUN_ID', env)

    def test_failed_child_is_reported(self):
        child = MagicMock()
        child.poll.return_value = 1
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch('torch.cuda.is_available', return_value=True), \
                     patch('torch.cuda.device_count', return_value=2), \
                     patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1'}), \
                     patch('src.training.grid_search.subprocess.Popen', return_value=child):
                    with self.assertRaisesRegex(RuntimeError, '1 trial'):
                        run_grid_search({'model': 'bird_cnn_delayed_pooling'}, [0, 1],
                                        120, [0.001], [0.1])
            finally:
                os.chdir(cwd)

    def test_interrupt_stops_children(self):
        children = [MagicMock(), MagicMock()]
        for child in children:
            child.poll.return_value = None
        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch('torch.cuda.is_available', return_value=True), \
                     patch('torch.cuda.device_count', return_value=2), \
                     patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1'}), \
                     patch('src.training.grid_search.subprocess.Popen', side_effect=children), \
                     patch('src.training.grid_search.time.sleep', side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        run_grid_search({'model': 'bird_cnn_delayed_pooling'}, [0, 1],
                                        120, [0.001], [0.05, 0.1])
                for child in children:
                    child.terminate.assert_called_once()
                    child.wait.assert_called_once()
            finally:
                os.chdir(cwd)

    def test_preview_does_not_launch_or_require_cuda(self):
        with patch('src.training.grid_search.subprocess.Popen') as launch, \
             patch('torch.cuda.is_available', return_value=False):
            run_grid_search({'model': 'bird_cnn_delayed_pooling'}, [0, 1], 120,
                            [0.001], [0.1], dry_run=True)
            launch.assert_not_called()

    def test_missing_gpu_and_invalid_parameters_fail_before_launch(self):
        config = {'model': 'bird_cnn_delayed_pooling'}
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaises(ValueError):
                run_grid_search(config, [0, 1], 120, [0.001], [0.1])
        for gpus, epochs, rates, dropouts in [([0, 0], 120, [0.001], [0.1]),
                                             ([0, 1], 0, [0.001], [0.1]),
                                             ([0, 1], 120, [float('nan')], [0.1])]:
            with self.assertRaises(ValueError):
                run_grid_search(config, gpus, epochs, rates, dropouts, dry_run=True)


if __name__ == '__main__':
    unittest.main()
