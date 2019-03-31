import numpy as np
from sklearn.manifold import TSNE
from matplotlib.pyplot as plt

def tSNE_plot(X,y):
    X_compressed_2d = TSNE(n_components=2).fit_transform(X)
    (fig, subplots) = plt.subplots(1, 1, figsize=(15, 8))
    ax = subplots[0][0]