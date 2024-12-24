# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from sklearn.decomposition import PCA
pca_time = PCA(n_components=20)
pca_space = PCA(n_components=16)
df  =np.load("./data/cylinder.npy")
print(df.shape)
pca_time.fit(df)
pca_time_components = np.ascontiguousarray(pca_time.components_.T)
pca_space.fit(pca_time_components)
space_time_basis = pca_space.components_

print(pca_time_components.shape)
print(space_time_basis.shape)