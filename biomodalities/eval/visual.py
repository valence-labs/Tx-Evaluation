import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.decomposition import PCA
import math

class OfflineVIZ:
    def __init__(self, color_palette: str = "tab20"):
        """Offline visualization helper for UMAP and PCA.

        Args:
            color_palette (str, optional): Color scheme for the classes. Defaults to "hls".
        """
        self.color_palette = color_palette

    def plot(self, embeddings: np.ndarray, labels: np.ndarray):
        """Produces UMAP and PCA visualizations using precomputed embeddings and labels.

        Args:
            embeddings (np.ndarray): Embeddings array where each row is a data point.
            labels (np.ndarray): Corresponding labels array.

        Returns:
            tuple: Tuple containing matplotlib figure objects for UMAP and PCA.
        """
        num_classes = len(np.unique(labels))

        # UMAP Plot
        print("Creating UMAP")
        umap_data = umap.UMAP(n_components=2).fit_transform(embeddings)
        umap_df = pd.DataFrame(umap_data, columns=["UMAP-1", "UMAP-2"])
        umap_df['Y'] = labels

        fig_umap = plt.figure(figsize=(12.6, 12.6))  # Enlarged by 40%
        ax_umap = sns.scatterplot(
            x="UMAP-1", y="UMAP-2", hue="Y",
            palette=sns.color_palette(self.color_palette, num_classes),
            data=umap_df, legend='brief', alpha=0.6, s=25, marker='o'  # Updated marker style and size
        )
        ax_umap.set(xlabel='', ylabel='', xticklabels=[], yticklabels=[])
        ax_umap.tick_params(left=False, right=False, bottom=False, top=False)
        plt.legend(loc="best", bbox_to_anchor=(1.05, 0.95), borderaxespad=0.)
        plt.tight_layout()

        # PCA Plot
        print("Creating PCA")
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(embeddings)
        pca_df = pd.DataFrame(pca_data, columns=["PCA-1", "PCA-2"])
        pca_df['Y'] = labels

        explained_var = pca.explained_variance_ratio_
        total_var = explained_var.sum() * 100
        labels_pca = [f"PCA-1 ({explained_var[0]*100:.2f}%)", f"PCA-2 ({explained_var[1]*100:.2f}%)"]
        title_pca = f"Total Explained Variance: {total_var:.2f}%"

        fig_pca = plt.figure(figsize=(12.6, 12.6))
        ax_pca = sns.scatterplot(
            x="PCA-1", y="PCA-2", hue="Y",
            palette=sns.color_palette(self.color_palette, num_classes),
            data=pca_df, legend='brief', alpha=0.6, s=50, marker='o'
        )
        ax_pca.set_xlabel(labels_pca[0], fontsize=14)
        ax_pca.set_ylabel(labels_pca[1], fontsize=14)
        ax_pca.set_xticklabels([])
        ax_pca.set_yticklabels([])
        ax_pca.tick_params(left=False, right=False, bottom=False, top=False)
        plt.title(title_pca)
        plt.legend(loc="best", bbox_to_anchor=(1.05, 0.95), borderaxespad=0.)
        plt.tight_layout()

        return fig_umap, fig_pca


if __name__ == '__main__' :
    embeddings = np.random.rand(1000, 128)  # Example embeddings
    labels = np.random.randint(0, 20, 1000)  # Example labels for 20 classes
    

    visualization = OfflineVIZ(color_palette="tab20")
    fig_umap, fig_pca = visualization.plot(embeddings, labels)

    # Save each fig in separate file
    fig_umap.savefig("umap.png")
    fig_pca.savefig("pca.png")
