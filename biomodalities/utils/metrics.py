from typing import Dict, List, Sequence

import torch
import torchmetrics
import torch.nn.functional as F

import numpy as np
import logging
import tqdm

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_reconstruction_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Computes various metrics to evaluate reconstruction quality including MSE, MAE, Pearson correlation,
    R-squared, and Wasserstein distance.
    """
    logging.info("Function called: evaluate_reconstruction_metrics")
    logging.info(f"Device of predictions: {predictions.device}, Device of targets: {targets.device}")

    # Ensure predictions and targets are on the same device
    targets = targets.to(predictions.device)
    logging.info(f"Shapes - Predictions: {predictions.shape}, Targets: {targets.shape}")

    # Initialize metrics
    mse_metric = torchmetrics.MeanSquaredError().to(predictions.device)
    mae_metric = torchmetrics.MeanAbsoluteError().to(predictions.device)
    pearson_metric = torchmetrics.PearsonCorrCoef(num_outputs=predictions.shape[1]).to(predictions.device)
    spearman_metric = torchmetrics.SpearmanCorrCoef(num_outputs=predictions.shape[1]).to(predictions.device)

    # Compute MSE and MAE
    mse = mse_metric(predictions, targets).item()
    mae = mae_metric(predictions, targets).item()
    logging.info(f"MSE: {mse}, MAE: {mae}")

    # Calculate Pearson correlation for each feature and average them
    correlations = []
    # Calculate Pearson and Spearman correlations
    average_pearson = pearson_metric(predictions, targets).mean().item()
    average_spearman = spearman_metric(predictions, targets).mean().item()
    logging.info(f"Average Pearson Correlation: {average_pearson}")
    logging.info(f"Average Spearman Correlation: {average_spearman}")

    # R-squared calculation
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    r_squared = r_squared.item()
    logging.info(f"R-squared: {r_squared}")


    # Return all metrics as a dictionary
    metrics = {
        "MSE": mse,
        "MAE": mae,
        "Average Pearson Correlation": average_pearson,
        "Average Spearman Correlation": average_spearman,
        "R-squared": r_squared
    }
    logging.info(f"Computed Metrics: {metrics}")
    mse_metric.reset()
    mae_metric.reset()
    pearson_metric.reset()
    spearman_metric.reset()

    return metrics

def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5)
) -> Sequence[int]:
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        outputs (torch.Tensor): output of a classifier (logits or probabilities).
        targets (torch.Tensor): ground truth labels.
        top_k (Sequence[int], optional): sequence of top k values to compute the accuracy over.
            Defaults to (1, 5).

    Returns:
        Sequence[int]:  accuracies at the desired k.
    """

    with torch.no_grad():
        maxk = max(top_k)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.

    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.
        batch_size_key (str): key of batch size values.

    Returns:
        float: weighted mean of the values of a key
    """

    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value.squeeze(0)



class StructuralTranscriptomeDistance(torchmetrics.Metric):
    def __init__(self, distance_function='frobenius', compute_on_cpu=False):
        super().__init__(compute_on_cpu=compute_on_cpu)
        self.distance_function = distance_function
        self.add_state("total_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_max_distance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_elements", default=torch.tensor(0), dist_reduce_fx="sum")
        # Store control means per batch
        self.control_means = {}

    def adjust_data(self, transcriptomes, batch_ids, is_control, control_label):
        # Convert control information to a boolean mask
        if not isinstance(is_control, np.ndarray):
            is_control = np.array(is_control)
        control_mask = is_control == control_label

        # Convert batch IDs to numpy for indexing
        if isinstance(batch_ids, torch.Tensor):
            batch_ids = batch_ids.numpy()
        
        unique_batches = np.unique(batch_ids)
        adjusted_transcriptomes = transcriptomes.clone()
        
        for batch in unique_batches:
            batch_mask = batch_ids == batch
            control_batch_mask = batch_mask & control_mask

            if np.any(control_batch_mask):
                mean_control = transcriptomes[torch.from_numpy(control_batch_mask)].mean(dim=0)
                adjusted_transcriptomes[torch.from_numpy(batch_mask)] -= mean_control

        return adjusted_transcriptomes

    

    def update(self, preds, target, batch_ids, is_control, control_label):
        # Adjust all data first
        pred_adjusted = self.adjust_data(preds, batch_ids, is_control, control_label)
        target_adjusted = self.adjust_data(target, batch_ids, is_control, control_label)
        
        # Convert batch IDs to numpy if necessary for indexing
        if isinstance(batch_ids, torch.Tensor):
            batch_ids = batch_ids.numpy()
        
        # Get unique batch IDs
        unique_batches = np.unique(batch_ids)
        
        # Compute distances for each batch
        for batch in tqdm.tqdm(unique_batches):
            # Create a mask for the current batch
            batch_mask = batch_ids == batch
            
            # Select the corresponding data for this batch
            pred_batch = pred_adjusted[torch.from_numpy(batch_mask)]
            target_batch = target_adjusted[torch.from_numpy(batch_mask)]
            
            if len(pred_batch) > 0:  # Ensure that there are elements in the batch
                if self.distance_function == 'frobenius':
                    #logging.info(f"\ntarget batch shape: {target_batch.shape}")
                    # Calculate the Frobenius distance for this batch
                    distance = torch.norm(pred_batch - target_batch, p='fro') / target_batch.shape[0]
                    max_distance = torch.norm(target_batch - 0, p='fro') / target_batch.shape[0]
                    
                    # Accumulate distances and count the number of batches
                    self.total_distance += distance
                    self.total_max_distance += max_distance
                    self.num_elements += 1

    def compute(self):
        # Calculate average distance
        distance = self.total_distance / self.num_elements if self.num_elements > 0 else self.total_distance
        max_distance = 2 * (self.total_max_distance / self.num_elements) if self.num_elements > 0 else  2 * self.total_max_distance
        integrity = 1 - (distance /  max_distance)
        return integrity.item()



if __name__ == "__main__":
    # Test accuracy_at_k
    outputs = torch.tensor([[0.2, 0.8, 0.1], [0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])
    targets = torch.tensor([1, 0, 2])
    top_k = (1, 2, 3)
    accuracies = accuracy_at_k(outputs, targets, top_k)
    accuracies = [acc.item() for acc in accuracies]  # Convert tensors to floats for printing
    print("Accuracies at top k:", accuracies)

    # Test weighted_mean
    outputs_list = [
        {"accuracy": torch.tensor(0.8), "batch_size": torch.tensor(5)},
        {"accuracy": torch.tensor(0.9), "batch_size": torch.tensor(10)},
        {"accuracy": torch.tensor(0.75), "batch_size": torch.tensor(15)}
    ]
    key = "accuracy"
    batch_size_key = "batch_size"
    mean_accuracy = weighted_mean(outputs_list, key, batch_size_key)
    mean_accuracy = mean_accuracy  # Convert tensor to float for printing
    print("Weighted mean accuracy:", mean_accuracy)


    # Generate synthetic data for testing with multiple features
    torch.manual_seed(0)  # For reproducibility
    num_samples = 100
    num_features = 20  # Simulate a scenario with 20 features
    predictions = torch.rand(num_samples, num_features)
    targets = torch.rand(num_samples, num_features) + 0.5  # Offset to create a difference

    # Compute metrics
    metrics = evaluate_reconstruction_metrics(predictions, targets)
    print("Reconstruction Metrics:", metrics)
    

    # Seed for reproducibility
    torch.manual_seed(42)

    # Generate synthetic prediction and target data
    num_samples = 10
    num_features = 5
    preds = torch.randn(num_samples, num_features)
    targets = torch.randn(num_samples, num_features)

    # Batch and control data
    batch_ids = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    controls = np.array(["non-targeting", "a", "non-targeting", "b", "c", "non-targeting", "non-targeting", "d", "z", "g"])

    # Initialize the structural distance metric
    structural_distance = StructuralTranscriptomeDistance()

    # Update the metric with the synthetic data
    structural_distance.update(preds, targets, batch_ids, controls)

    # Compute the final distance
    final_distance = structural_distance.compute()

    print("Structural reconstruction distance:", final_distance)
