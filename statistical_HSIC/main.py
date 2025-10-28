import sys
sys.path.insert(0,'./')
from statistical_HSIC.utils import hsic_embedding
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import glob, os
import umap

def main():
    n_samples = 2000
    for folder in ['MNIST_Augmentations_1', 'MNIST_Augmentations_2']:
        paths = np.sort(glob.glob(os.path.join('./', folder, '*.png')))
        # X = np.zeros((len(paths), 28*28))
        # y = np.zeros((len(paths),))
        X = np.zeros((n_samples, 28*28))
        y = np.zeros((n_samples,))
        for i, path in enumerate(paths):
            if i >= n_samples:
                break
            label = int(str(path).split('.')[-2][-1])
            y[i] = label
            image = Image.open(path)
            image_array = np.array(image)
            X[i, :] = image_array.reshape(-1,)
        if folder == 'MNIST_Augmentations_1':
            X1 = X.copy()
            y1 = y.copy()
        else:
            X2 = X.copy()
            y2 = y.copy()

    # scaler = StandardScaler()
    # X = scaler.fit_transform(np.vstack((X1, X2)))
    # X1, X2 = X[:X1.shape[0], :], X[X1.shape[0]:, :]

    mapper = hsic_embedding.HSIC_Embedding(n_components=10, kernel_on_labels='rbf')
    X1_transformed, X2_transformed = mapper.fit_transform(X1=X1, X2=X2)

    umap_model = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_model.fit_transform(np.vstack((X1_transformed, X2_transformed)))
    X1_transformed, X2_transformed = X_umap[:X1_transformed.shape[0], :], X_umap[X1_transformed.shape[0]:, :]

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        X1_transformed[:, 0], X1_transformed[:, 1], 
        c=y1, cmap='tab10', alpha=0.7, marker='o', s=10
    )
    scatter = plt.scatter(
        X2_transformed[:, 0], X2_transformed[:, 1], 
        c=y2, cmap='tab10', alpha=0.7, marker='o', s=10
    )

    # Add a color bar with label ticks
    cbar = plt.colorbar(scatter)
    cbar.set_label('Digit Label')
    plt.clim(-0.5, 9.5)  # Ensure labels align with integers 0-9

    # Add labels and title
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Projected data')
    plt.grid(alpha=0.3)
    plt.show()

if __name__ == '__main__':
    main()