"""Test inference and test-only preprocessing using synthetic data, without remote access."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from datasets import Dataset, Features, ClassLabel, Image as HFImage, Value
from PIL import Image
import torch
from torch import nn

from src.data.dataset import make_test_loader
from src.evaluation.test_evaluation import download_run_weights, evaluate_test, log_test_results


class TestEvaluationTests(unittest.TestCase):
    def test_test_split_bbox_crop_then_resize_without_augmentation(self):
        # The left half is the bird box; the right half must disappear.
        image = Image.new('RGB', (12, 6), (0, 0, 255))
        image.paste((255, 0, 0), (0, 0, 6, 6))
        dataset = Dataset.from_dict(
            {'image': [image], 'label': [0], 'file_name': ['bird.jpg']},
            features=Features({'image': HFImage(), 'label': ClassLabel(names=['bird']),
                               'file_name': Value('string')}),
        )
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'images.txt').write_text('1 species/bird.jpg\n')
            Path(directory, 'bounding_boxes.txt').write_text('1 0 0 6 6\n')
            config = dict(image_size=8, batch_size=32, use_bounding_boxes=True,
                          bbox_annotation_dir=directory, random_crop_probability=1,
                          cutout_probability=1, horizontal_flip_probability=1)
            with patch('datasets.load_dataset', return_value=dataset) as load:
                loader, names = make_test_loader(config)
                first = loader.dataset[0]['image']
                second = loader.dataset[0]['image']
            load.assert_called_once_with('Multimodal-Fatima/CUB_test', split='test')
            self.assertEqual(first.shape, (3, 8, 8))
            self.assertTrue(torch.equal(first, second))
            self.assertTrue(torch.all(first[0] == 1))
            self.assertTrue(torch.all(first[1:] == 0))
            self.assertEqual(names, ['bird'])
            self.assertEqual(loader.num_workers, 2)

    def test_inference_uses_eval_no_grad_and_sample_weighted_loss(self):
        model = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout(0.9))
        before = {key: value.clone() for key, value in model.state_dict().items()}
        inputs = torch.tensor([[4., 0.], [0., 4.], [0., 4.]])
        targets = torch.tensor([0, 1, 0])
        batches = [{'image': inputs[:2], 'label': targets[:2]},
                   {'image': inputs[2:], 'label': targets[2:]}]
        metrics, report, count = evaluate_test(model, batches, ['a', 'b'], torch.device('cpu'))
        self.assertEqual(count, 3)
        self.assertAlmostEqual(metrics['accuracy'], 200 / 3)
        self.assertAlmostEqual(metrics['macro_recall'], 75)
        expected = nn.functional.cross_entropy(model(inputs), targets).item()
        self.assertAlmostEqual(metrics['loss'], expected, places=5)
        self.assertIn('precision', report)
        self.assertFalse(model.training)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_download_selects_single_artifact_and_restores_inference_settings(self):
        run = SimpleNamespace(data=SimpleNamespace(params={
            'model': 'bird_cnn_delayed_pooling', 'image_size': '320',
            'use_bounding_boxes': 'True', 'dropout2d': '0.075',
        }), info=SimpleNamespace(experiment_id='7'))
        client = MagicMock()
        client.get_run.return_value = run
        client.list_artifacts.return_value = [SimpleNamespace(path='weights/trial.pth', is_dir=False)]
        config = dict(model='bird_cnn', image_size=224, use_bounding_boxes=False,
                      bbox_annotation_dir='local/annotations')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, 'trial.pth')
            torch.save({}, path)
            with patch('src.evaluation.test_evaluation.MlflowClient', return_value=client), \
                 patch('mlflow.set_tracking_uri'), \
                 patch('mlflow.artifacts.download_artifacts', return_value=str(path)) as download:
                effective, weights, experiment, artifact = download_run_weights(config, 'run123')
            self.assertEqual(weights, str(path))
            self.assertEqual(experiment, '7')
            self.assertEqual(artifact, 'weights/trial.pth')
            self.assertEqual(download.call_args.kwargs['run_id'], 'run123')
        self.assertEqual(config['image_size'], 224)
        self.assertEqual(effective['image_size'], 320)
        self.assertTrue(effective['use_bounding_boxes'])
        self.assertEqual(effective['bbox_annotation_dir'], 'local/annotations')

    def test_ambiguous_weights_require_explicit_selection(self):
        client = MagicMock()
        client.get_run.return_value = SimpleNamespace(data=SimpleNamespace(params={}))
        client.list_artifacts.return_value = []
        with patch('src.evaluation.test_evaluation.MlflowClient', return_value=client), \
             patch('mlflow.set_tracking_uri'):
            with self.assertRaisesRegex(ValueError, 'artifact-path'):
                download_run_weights({'model': 'bird_cnn'}, 'missing')

    def test_cli_uses_only_test_loader_and_loads_saved_weights(self):
        import yaml
        from src import cli

        config = dict(model='bird_cnn', num_classes=2, dropout=0, seed=42,
                      image_size=8, batch_size=2, use_bounding_boxes=True)
        model = nn.Linear(2, 2)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory, 'config.yaml')
            config_path.write_text(yaml.safe_dump(config))
            weights = Path(directory, 'weights.pth')
            torch.save(model.state_dict(), weights)
            with patch('sys.argv', ['cli', 'test', '--config', str(config_path), '--run-id', 'source']), \
                 patch('src.evaluation.test_evaluation.download_run_weights',
                       return_value=(config, str(weights), '7', 'weights/model.pth')), \
                 patch.object(cli, 'BirdCNN', return_value=model), \
                 patch.object(cli, 'make_loaders', side_effect=AssertionError('Loaded training data')), \
                 patch('src.data.dataset.make_test_loader', return_value=([], ['a', 'b'])) as loader, \
                 patch('src.evaluation.test_evaluation.evaluate_test',
                       return_value=({'accuracy': 50}, 'report', 2)) as evaluate, \
                 patch('src.evaluation.test_evaluation.log_test_results') as log:
                cli.main()
        loader.assert_called_once_with(config)
        evaluate.assert_called_once()
        log.assert_called_once()

    def test_results_go_to_separate_evaluation_run(self):
        with patch('mlflow.start_run') as start, patch('mlflow.set_tags') as tags, \
             patch('mlflow.log_params'), patch('mlflow.log_metrics') as log_metrics, \
             patch('mlflow.log_text') as log_text:
            log_test_results({'accuracy': 50, 'loss': 2, 'macro_recall': 50}, 'report', 10,
                             {'model': 'bird_cnn', 'image_size': 320, 'batch_size': 32},
                             'source123', '7', 'weights/trial.pth')
        self.assertEqual(start.call_args.kwargs['experiment_id'], '7')
        self.assertNotIn('run_id', start.call_args.kwargs)
        self.assertEqual(tags.call_args.args[0]['source_run_id'], 'source123')
        self.assertEqual(log_metrics.call_args.args[0]['test_accuracy'], 50)
        log_text.assert_called_once_with('report', 'test/classification_report.txt')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
