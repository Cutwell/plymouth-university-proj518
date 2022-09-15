import numpy as np
import matplotlib.pyplot as plt
import scipy.stats.stats as st
import scipy.spatial.distance as spd
import sklearn.decomposition as skd
import sklearn.manifold as skm
import matplotlib.patches as mpatches

def unpack_platypus(optimiser):
    """
    Take the Pareto front approximation from a Platypus optimiser and
    return a tuple of Numpy arrays -- one holds the decision variables and
    the other hold the objective variables.
    """
    X = np.array([soln.variables for soln in optimiser.result])
    Y = np.array([soln.objectives for soln in optimiser.result])
    return X, Y


def rank_coords(Y):
    """
    Convert the objective vectors to rank coordinates.
    """
    N, M = Y.shape
    R = np.zeros((N, M))

    for m in range(M):
        R[:,m] = st.rankdata(Y[:,m])

    return R


def rank_best_obj(Y):
    """
    Rank the given objective vectors according to the objective on which 
    they have the best rank.
    """
    R = rank_coords(Y)
    return R.argmin(axis=1).astype(np.int)


def average_rank(Y):
    """
    Rank the solutions according to their average rank score.
    """
    R = rank_coords(Y)
    return R.mean(axis=1)


def parallel_coords(Y, colours=None, cmap="viridis", xlabels=None):
    """
    Produce a parallel coordinate plot for the objective space provided.
    """
    plt.figure()
    plt.title("Parallel Coordinates Plot")
    N, M = Y.shape

    if colours is None:
        colours = ["k"] * N         # Not really ideal, needs fixing.

    objTicks = np.arange(M, dtype=np.int)
    if xlabels is None:
        xlabels = objTicks + 1
    
    for i in range(N):
        plt.plot(objTicks, Y[i], c=colours[i])

    plt.xticks(objTicks, xlabels)
    plt.xlabel("Objective")
    plt.ylabel("$f(\mathbf{x})$")

    plt.show()


def scatter_plot(Z, colours=None, cmap="virids", title="Plot", labels=None):
    """
    Produce a scatter plot with the embedding provided.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.title(title)
    N = Z.shape[0]

    if colours is None:
        colours = ["k"] * N         # Not really ideal, needs fixing.

    # Plot the transformed solutions.
    plt.scatter(Z[:,0], Z[:,1], c=colours, cmap=cmap)

    # Tidy up the plot.
    #plt.xticks([])
    #plt.yticks([])

    plt.show()


def pca_projection(Y, colours=None, cmap="viridis", labels=None):
    """
    Project the points into two dimensions with PCA, and produce
    a corresponding plot.
    """
    # Perform the PCA projection.
    pca = skd.PCA(n_components=2)
    Z = pca.fit_transform(Y)
    scatter_plot(Z, colours=colours, cmap=cmap, title="PCA Projection", labels=labels)


def mds_projection(Y, metric="euclidean", colours=None, cmap="viridis", labels=None):
    """
    Project the points into two dimensions with MDS, and produce
    a corresponding plot. Use either Euclidean distance or dominance 
    distance.
    """
    if not metric in ["euclidean", "dominance"]:
        raise RuntimeError("Unknown metric - expected 'euclidean' or 'dominance'")

    N, M = Y.shape

    if metric == "euclidean":
        mds = skm.MDS(n_components=2)
        Z = mds.fit_transform(Y)
        
    else:
        # Compute the dominance distance.
        R = rank_coords(Y)
        D = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                D[i,j] = (1/M) * (abs(R[i] - R[j])).sum()

        mds = skm.MDS(n_components=2, dissimilarity="precomputed")
        Z = mds.fit_transform(D)

    scatter_plot(Z, colours=colours, cmap=cmap, title="MDS Projection, metric={}".format(metric), labels=labels)