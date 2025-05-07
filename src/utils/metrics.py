import torch
import torch.nn.functional as F

class DistributionMetrics:
    """
    Collection of metrics for comparing probability distributions.
    All methods expect input tensors of shape [batch_size, num_points].
    """

    @staticmethod
    def kl_divergence(p, q, epsilon=1e-10):
        """
        Compute Kullback-Leibler divergence between p and q.

        Args:
            p (torch.Tensor): First distribution
            q (torch.Tensor): Second distribution
            epsilon (float): Small constant for numerical stability

        Returns:
            torch.Tensor: KL(P||Q) for each sample in the batch
        """
        # Ensure valid probability distributions
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)

        # Add small epsilon to avoid log(0)
        p = p + epsilon
        q = q + epsilon

        # Normalize
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)

        return torch.sum(p * torch.log(p / q), dim=-1)

    @staticmethod
    def js_divergence(p, q, epsilon=1e-10):
        """
        Compute Jensen-Shannon divergence between p and q.

        Args:
            p (torch.Tensor): First distribution
            q (torch.Tensor): Second distribution
            epsilon (float): Small constant for numerical stability

        Returns:
            torch.Tensor: JS(P||Q) for each sample in the batch
        """
        # Ensure valid probability distributions
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)

        # Add small epsilon to avoid log(0)
        p = p + epsilon
        q = q + epsilon

        # Normalize
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)

        # Compute midpoint distribution
        m = 0.5 * (p + q)

        # JS = 0.5 * (KL(P||M) + KL(Q||M))
        js_p = torch.sum(p * torch.log(p / m), dim=-1)
        js_q = torch.sum(q * torch.log(q / m), dim=-1)

        return 0.5 * (js_p + js_q)

    @staticmethod
    def wasserstein_distance(p, q):
        """
        Compute Wasserstein distance between p and q using their empirical CDFs.

        Args:
            p (torch.Tensor): First distribution
            q (torch.Tensor): Second distribution

        Returns:
            torch.Tensor: Wasserstein distance for each sample in the batch
        """
        # Compute empirical CDFs
        p_cdf = torch.cumsum(F.softmax(p, dim=-1), dim=-1)
        q_cdf = torch.cumsum(F.softmax(q, dim=-1), dim=-1)

        # Return L1 distance between CDFs
        return torch.sum(torch.abs(p_cdf - q_cdf), dim=-1)

    @staticmethod
    def compute_all_metrics(p, q):
        """
        Compute all available metrics between p and q.

        Args:
            p (torch.Tensor): First distribution
            q (torch.Tensor): Second distribution

        Returns:
            dict: Dictionary containing all computed metrics
        """
        return {
            'kl_divergence': DistributionMetrics.kl_divergence(p, q).mean().item(),
            'js_divergence': DistributionMetrics.js_divergence(p, q).mean().item(),
            'wasserstein': DistributionMetrics.wasserstein_distance(p, q).mean().item()
        }

def evaluate_generated_samples(generated, target):
    """
    Evaluate generated samples against target samples using multiple metrics.

    Args:
        generated (torch.Tensor): Generated samples [batch_size, num_points]
        target (torch.Tensor): Target samples [batch_size, num_points]

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    return DistributionMetrics.compute_all_metrics(generated, target)
