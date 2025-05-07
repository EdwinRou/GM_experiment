import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import generate_mixture_data, MixtureDataset, get_dataloaders

class TestDataGeneration(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.num_samples = 10
        self.num_points = 100
        self.range_min = -5
        self.range_max = 5

    def test_generate_mixture_data(self):
        """Test mixture data generation with default parameters."""
        samples, x_values = generate_mixture_data(
            num_samples=self.num_samples,
            num_points=self.num_points,
            range_min=self.range_min,
            range_max=self.range_max
        )

        # Check shapes
        self.assertEqual(samples.shape, (self.num_samples, self.num_points))
        self.assertEqual(x_values.shape, (self.num_points,))

        # Check value ranges
        self.assertTrue(torch.all(samples >= -1))
        self.assertTrue(torch.all(samples <= 1))
        self.assertTrue(torch.all(x_values >= self.range_min))
        self.assertTrue(torch.all(x_values <= self.range_max))

    def test_mixture_dataset(self):
        """Test MixtureDataset class."""
        data = torch.randn(self.num_samples, self.num_points)
        dataset = MixtureDataset(data)

        # Check length
        self.assertEqual(len(dataset), self.num_samples)

        # Check item retrieval
        item = dataset[0]
        self.assertEqual(item.shape, (self.num_points,))
        torch.testing.assert_close(item, data[0])

    def test_get_dataloaders(self):
        """Test dataloader creation."""
        batch_size = 4
        train_loader, test_loader, x_coords = get_dataloaders(
            batch_size=batch_size,
            num_train=8,
            num_test=4,
            seed=42
        )

        # Check batch sizes
        train_batch = next(iter(train_loader))
        self.assertEqual(train_batch.shape, (batch_size, self.num_points))

        test_batch = next(iter(test_loader))
        self.assertEqual(test_batch.shape, (batch_size, self.num_points))

        # Check x_coords
        self.assertEqual(x_coords.shape, (self.num_points,))

if __name__ == '__main__':
    unittest.main()
