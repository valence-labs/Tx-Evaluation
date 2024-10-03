import tqdm
import torch
import torchmetrics
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from typing import List


class TorchILISIMetric(torchmetrics.Metric):
    """
    A custom TorchMetrics class for calculating the Inverse Local Simpson's Index (ILISI),
    designed to measure the diversity within local neighborhoods in high-dimensional data spaces.
    This metric helps assess how well different data batches or conditions are integrated.
    """
    def __init__(self, unique_labels: List[str], perplexity: int = 30, dist_sync_on_step: bool = False, use_pca: bool = False):
        """
        Initializes the metric with necessary parameters and state variables.
        
        Args:
            unique_labels (List[str]): A list of unique categorical labels in the dataset.
            perplexity (int): Number of nearest neighbors to use for calculating the diversity index.
            dist_sync_on_step (bool): If true, synchronizes metric state across devices during training.
            use_pca (bool): If True, applies PCA to reduce dimensions to the first 32 principal components.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_cpu=True)
        self.perplexity = perplexity
        self.use_pca = use_pca
        self.unique_labels = unique_labels
        # Map each unique label to an index for numerical processing.
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        # State variables to accumulate embeddings and their corresponding labels across batches.
        self.add_state("embeddings", default=[], dist_reduce_fx=None)
        self.add_state("labels", default=[], dist_reduce_fx=None)  

    def _extract_labels(self, labels):
        """
        Converts a list of categorical labels to their corresponding indices based on pre-defined mapping.
        
        Args:
            labels: A list or array of labels to convert.
        
        Returns:
            Torch tensor of label indices.
        """
        label_indices = torch.tensor([self.label_to_idx[label] for label in labels], dtype=torch.long)
        return label_indices

    def update(self, embeddings: torch.Tensor, metadata: pd.Series):
        """
        Updates the states (embeddings and labels) with new data as it comes in batches.
        
        Args:
            embeddings (torch.Tensor): The embeddings tensor for the current batch.
            metadata (pd.Series): The series containing labels corresponding to the embeddings.
        """
        # Extract label indices from metadata. If metadata is already a tensor, use it directly.
        label_indices = self._extract_labels(metadata) if not isinstance(metadata, torch.Tensor) else metadata
        # Detach embeddings from the computation graph and store them in the state list.
        self.embeddings.append(embeddings.detach().cpu())
        # Append the processed label indices to the state list.
        self.labels.append(label_indices)

    def compute(self) -> torch.Tensor:
        """
        Computes the final normalized ILISI score by processing all stored embeddings and labels.
        
        Returns:
            A single torch.Tensor representing the mean normalized ILISI score.
        """
        # Concatenate all stored embeddings and labels into single tensors.
        all_embeddings = torch.cat(self.embeddings, dim=0).numpy()
        all_labels = torch.cat(self.labels, dim=0).numpy()

        # Apply PCA if flagged true, reducing dimensions to the first 32 principal components.
        if self.use_pca:
            pca = PCA().fit(all_embeddings)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.where(cumulative_variance >= 0.9)[0][0] + 1
            print("n_components : ",n_components)
            n_components = max(32,n_components)
            pca = PCA(n_components=n_components)
            all_embeddings = pca.fit_transform(all_embeddings)
            print(f"Sum of explained variance for the top {n_components} components: {sum(pca.explained_variance_ratio_[:n_components]) * 100:.2f}%")


        # Calculate ILISI scores using a single label computation function.
        lisi_scores = self._compute_lisi_single_label(all_embeddings, all_labels, self.perplexity)
        # Normalize the LISI scores to the range [1, 2]
        k = self.perplexity * 3
        normalized_lisi_scores = 1 + (np.array(lisi_scores) - 1) / (k - 1)
        # Calculate the mean of normalized scores and return.
        return torch.mean(torch.tensor(lisi_scores, dtype=torch.float32))


    @staticmethod
    def _compute_lisi_single_label(embeddings: np.ndarray, labels, perplexity: int = 30):
        """
        Helper function to compute ILISI scores for provided embeddings and labels.
        
        Args:
            embeddings (np.ndarray): Embeddings for which to calculate the scores.
            labels: Corresponding labels for the embeddings.
            perplexity (int): Number of neighbors to consider for neighborhood diversity calculation.
        
        Returns:
            Array of ILISI scores.
        """
        # Fit the nearest neighbors model and find neighbors for each point.
        nearest_neighbors = NearestNeighbors(n_neighbors=perplexity * 3, algorithm='kd_tree').fit(embeddings)
        distances, indices = nearest_neighbors.kneighbors(embeddings)
        # Exclude the nearest neighbor since it is the point itself.
        indices = indices[:, 1:] 
        distances = distances[:, 1:]

        # Convert labels to a categorical format for processing.
        labels = pd.Categorical(labels)
        num_categories = len(labels.categories)
        # Calculate Simpson indices for diversity assessment.
        simpson_indices = TorchILISIMetric._compute_simpson(distances.T, indices.T, labels, num_categories, perplexity)
        # Return the inverse of Simpson indices as ILISI scores.
        return 1 / simpson_indices

    @staticmethod
    def _compute_simpson(distances: np.ndarray, indices: np.ndarray, labels: pd.Categorical, num_categories: int, perplexity: float, tol: float = 1e-5):
        """
        Calculates the Simpson's index for each point based on its neighborhood.
        
        Args:
            distances (np.ndarray): Array of distances to the nearest neighbors.
            indices (np.ndarray): Array of indices of the nearest neighbors.
            labels (pd.Categorical): Categorical representation of labels.
            num_categories (int): Number of unique categories in the labels.
            perplexity (float): Number of neighbors to consider, which influences the entropy calculation.
            tol (float): Tolerance level for stopping the beta adjustment.
        
        Returns:
            Array of Simpson's indices for each point.
        """
        num_points = distances.shape[1]
        simpson_index = np.zeros(num_points)
        # Set initial target entropy based on perplexity.
        log_target_entropy = np.log(perplexity)

        # Iterate through each point to calculate its Simpson's index.
        for point_idx in tqdm.tqdm(range(num_points)):
            # Initialize beta parameters for entropy adjustment.
            beta, beta_min, beta_max = 1.0, -np.inf, np.inf
            # Start with initial probabilities based on distances.
            probabilities = np.exp(-distances[:, point_idx] * beta)
            prob_sum = np.sum(probabilities)

            if prob_sum == 0:
                entropy = 0
                probabilities = np.zeros(distances.shape[0])
            else:
                entropy = np.log(prob_sum) + beta * np.sum(distances[:, point_idx] * probabilities) / prob_sum
                probabilities /= prob_sum

            entropy_diff = entropy - log_target_entropy
            
            # Perform binary search to find the optimal beta that minimizes the difference between calculated and target entropy.
            for _ in range(50):
                if abs(entropy_diff) < tol:
                    break
                beta, beta_min, beta_max = TorchILISIMetric._adjust_beta(entropy_diff, beta, beta_min, beta_max)

                probabilities = np.exp(-distances[:, point_idx] * beta)
                prob_sum = np.sum(probabilities)

                if prob_sum == 0:
                    entropy = 0
                    probabilities = np.zeros(distances.shape[0])
                else:
                    entropy = np.log(prob_sum) + beta * np.sum(distances[:, point_idx] * probabilities) / prob_sum
                    probabilities /= prob_sum

                entropy_diff = entropy - log_target_entropy
            
            # Sum up the probabilities for each category to compute the Simpson's index.
            for category in labels.categories:
                category_indices = labels[indices[:, point_idx]] == category
                category_prob_sum = np.sum(probabilities[category_indices])
                simpson_index[point_idx] += category_prob_sum * category_prob_sum

        return simpson_index

    def _adjust_beta(entropy_diff, beta, beta_min, beta_max):
        """
        Adjusts the beta parameter based on the difference between calculated and target entropy.
        
        Args:
            entropy_diff (float): Difference between the current entropy and target entropy.
            beta (float): Current beta value.
            beta_min (float): Minimum value for beta.
            beta_max (float): Maximum value for beta.
        
        Returns:
            Updated values for beta, beta_min, and beta_max.
        """
        if entropy_diff > 0:
            beta_min = beta
            beta = beta * 2 if not np.isfinite(beta_max) else (beta + beta_max) / 2
        else:
            beta_max = beta
            beta = beta / 2 if not np.isfinite(beta_min) else (beta + beta_min) / 2
        return beta, beta_min, beta_max


if __name__ == "__main__":
    num_points = 3000
    num_features = 5
    metadata = pd.DataFrame({'batch': [f'batch{np.random.randint(1, 4)}' for _ in range(num_points)]})
    unique_labels = metadata['batch'].unique()
    for p in range(3,101, 2):
        # Instantiate the TorchILISIMetric
        ilisi_metric = TorchILISIMetric(perplexity=p,unique_labels=unique_labels)
        
        # Generate random embeddings
        embeddings = torch.randn(num_points, num_features)

        # Update the metric with synthetic data in batches
        batch_size = 50
        for i in range(0, num_points, batch_size):
            end = min(i + batch_size, num_points)
            ilisi_metric.update(embeddings[i:end], metadata['batch'].iloc[i:end])

        # Compute the ILISI score
        ilisi_score = ilisi_metric.compute()

        print("ILISI Score:", ilisi_score.item())