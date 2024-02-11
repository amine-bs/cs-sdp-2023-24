import pickle
from abc import abstractmethod

import numpy as np
from gurobipy import *
from sklearn.cluster import KMeans


class BaseModel(object):
    """
    Base class for models, to be used as coding pattern skeleton.
    Can be used for a model on a single cluster or on multiple clusters"""

    def __init__(self):
        """Initialization of your model and its hyper-parameters"""
        pass

    @abstractmethod
    def fit(self, X, Y):
        """Fit function to find the parameters according to (X, Y) data.
        (X, Y) formatting must be so that X[i] is preferred to Y[i] for all i.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # Customize what happens in the fit function
        return

    @abstractmethod
    def predict_utility(self, X):
        """Method to call the decision function of your model

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # Customize what happens in the predict utility function
        return

    def predict_preference(self, X, Y):
        """Method to predict which pair is preferred between X[i] and Y[i] for all i.
        Returns a preference for each cluster.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of preferences for each cluster. 1 if X is preferred to Y, 0 otherwise
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return (X_u - Y_u > 0).astype(int)

    def predict_cluster(self, X, Y):
        """Predict which cluster prefers X over Y THE MOST, meaning that if several cluster prefer X over Y, it will
        be assigned to the cluster showing the highest utility difference). The reversal is True if none of the clusters
        prefer X over Y.
        Compared to predict_preference, it indicates a cluster index.

        Parameters
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements to compare with Y elements of same index
        Y: np.ndarray
            (n_samples, n_features) list of features of elements to compare with X elements of same index

        Returns
        -------
        np.ndarray:
            (n_samples, ) index of cluster with highest preference difference between X and Y.
        """
        X_u = self.predict_utility(X)
        Y_u = self.predict_utility(Y)

        return np.argmax(X_u - Y_u, axis=1)

    def save_model(self, path):
        """Save the model in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the file in which the model will be saved
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(clf, path):
        """Load a model saved in a pickle file. Don't hesitate to change it in the child class if needed

        Parameters
        ----------
        path: str
            path indicating the path to the file to load
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class RandomExampleModel(BaseModel):
    """Example of a model on two clusters, drawing random coefficients.
    You can use it to understand how to write your own model and the data format that we are waiting for.
    This model does not work well but you should have the same data formatting with TwoClustersMIP.
    """

    def __init__(self):
        self.seed = 444
        self.weights = self.instantiate()

    def instantiate(self):
        """No particular instantiation"""
        return []

    def fit(self, X, Y):
        """fit function, sets random weights for each cluster. Totally independant from X & Y.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        np.random.seed(self.seed)
        num_features = X.shape[1]
        weights_1 = np.random.rand(num_features)  # Weights cluster 1
        weights_2 = np.random.rand(num_features)  # Weights cluster 2

        weights_1 = weights_1 / np.sum(weights_1)
        weights_2 = weights_2 / np.sum(weights_2)
        self.weights = [weights_1, weights_2]
        return self

    def predict_utility(self, X):
        """Simple utility function from random weights.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        u_1 = np.dot(X, self.weights[0])  # Utility for cluster 1 = X^T.w_1
        u_2 = np.dot(X, self.weights[1])  # Utility for cluster 2 = X^T.w_2
        return np.stack([u_1, u_2], axis=1)  # Stacking utilities over cluster on axis 1


class TwoClustersMIP(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces=5, n_clusters=2, n=4, PRECISION=1e-6, max_iterations=None):
        """Initialization of the MIP Variables

        Parameters
        ----------
        n_pieces: int
            Number of pieces for the utility function of each feature.
        n_clusters: int
            Number of clusters to implement in the MIP.
        """
        self.seed = 123
        self.model = self.instantiate()
        self.L = n_pieces
        self.K = n_clusters
        self.n = n
        self.PRECISION = PRECISION
        self.max_iterations = max_iterations

    def instantiate(self):
        """Instantiation of the MIP Variables - To be completed.""" # We cannot instantiate the MIP variables because we can only know how many variable we have once we have the training data.
        # To be completed

        return Model("MIP")

    def u_k_i(self, k, i, x, training=True):

        width = 1 / self.L
        l = int(x / width)
        if l == self.L:
            if training:
                return self.criteria[k-1][i-1][self.L]
            return self.criteria[k-1][i-1][self.L].x

        a = (x / width) - l
        if training:
            u_left = self.criteria[k-1][i-1][l]
            u_right = self.criteria[k-1][i-1][l+1]
        else:
            u_left = self.criteria[k-1][i-1][l].x
            u_right = self.criteria[k-1][i-1][l+1].x
        return u_left + a * (u_right - u_left)

    def u_k(self, k, x, training=True):
        if training:
            return quicksum(self.u_k_i(k, i, x[i-1]) for i in range(1, self.n + 1))
        else:
            return sum([self.u_k_i(k, i, x[i-1], training=False) for i in range(1, self.n + 1)])

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """

        np.random.seed(self.seed)

        self.criteria = [[[self.model.addVar(name=f"u_{k}_{i}_{l}") for l in range(self.L+1)] for i in range(1, self.n + 1)] for k in range(1, self.K + 1)]

        P = X.shape[0]
        self.sigma_plus = [self.model.addVar(name=f"sigma+_{j}", lb=0) for j in range(1, P + 1)]
        self.sigma_minus = [self.model.addVar(name=f"sigma-_{j}", lb=0) for j in range(1, P + 1)]

        self.pref = [[self.model.addVar(name=f"z_{k}_{j}", vtype=GRB.BINARY) for j in range(1, P + 1)] for k in range(1, self.K + 1)]

        for k in range(1, self.K + 1):
            for i in range(1, self.n + 1):
                self.model.addConstr(self.criteria[k-1][i-1][0] == 0)

        for k in range(1, self.K + 1):
            self.model.addConstr(quicksum(self.criteria[k-1][i-1][self.L] for i in range(1, self.n+1)) == 1)

        for k in range(1, self.K + 1):
            for i in range(1, self.n + 1):
                for l in range(0, self.L):
                    self.model.addConstr(self.criteria[k-1][i-1][l+1] - self.criteria[k-1][i-1][l] >= self.PRECISION)

        for j in range(1, P+1):
            self.model.addConstr(quicksum(self.pref[k-1][j-1] for k in range(1, self.K+1)) == 1)

        for j in range(1, P + 1): 
            x = X[j-1]
            y = Y[j-1]
            for k in range(1, self.K + 1):
                self.model.addConstr(self.u_k(k=k, x=x) - self.u_k(k=k, x=y) - self.sigma_minus[j-1] + self.sigma_plus[j-1] >= 2*(1 - self.pref[k-1][j-1]))
                self.model.addConstr(self.u_k(k=k, x=x) - self.u_k(k=k, x=y) + self.PRECISION <= 2*self.pref[k-1][j-1])

        self.model.setObjective(sum(self.sigma_plus) + sum(self.sigma_minus), GRB.MINIMIZE)      
        if self.max_iterations is not None:
            self.model.params.IterationLimit = self.max_iterations
        self.model.optimize()

        result = [[[self.criteria[k-1][i-1][l].x for l in range(0, self.L+1)] for i in range(1, self.n + 1)] for k in range(1, self.K+1)]
        pref = [[self.pref[k-1][j-1].x for j in range(1, P + 1)] for k in range(1, self.K + 1)]
        self.result_utility_ = np.array(result)
        self.result_pref_ = np.array(pref)
        return self

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.

        U = []
        for p in range(1, len(X)+1):
            utilities = [self.u_k(k, X[p-1], training=False) for k in range(1, self.K+1)]
            U.append(utilities)
        return np.array(U)


class HeuristicModel(BaseModel):
    """Skeleton of MIP you have to write as the first exercise.
    You have to encapsulate your code within this class that will be called for evaluation.
    """

    def __init__(self, n_pieces=5, n_clusters=3, n=10, PRECISION=1e-6, P=400):
        """Initialization of the Heuristic Model.
        """
        self.seed = 123
        self.L = n_pieces
        self.K = n_clusters
        self.n = n
        self.P = P
        self.PRECISION = PRECISION
        self.kmeans, self.models = self.instantiate()

    def instantiate(self):
        """Instantiation of the MIP Variables"""
        # To be completed
        kmeans = KMeans(n_clusters=self.K, random_state=self.seed)
        models = [TwoClustersMIP(self.L, 1, self.n, self.PRECISION) for k in range(self.K)]
        return kmeans, models

    def fit(self, X, Y):
        """Estimation of the parameters - To be completed.

        Parameters
        ----------
        X: np.ndarray
            (n_samples, n_features) features of elements preferred to Y elements
        Y: np.ndarray
            (n_samples, n_features) features of unchosen elements
        """
        # To be completed
        self.kmeans.fit(X - Y)
        Z_preds = self.kmeans.predict(X - Y)
        for k in range(self.K):
            P = min(self.P, X[Z_preds == k].shape[0])
            self.models[k] = self.models[k].fit(X[Z_preds == k][:P], Y[Z_preds == k][:P])
        return self

    def predict_utility(self, X):
        """Return Decision Function of the MIP for X. - To be completed.

        Parameters:
        -----------
        X: np.ndarray
            (n_samples, n_features) list of features of elements

        Returns
        -------
        np.ndarray:
            (n_samples, n_clusters) array of decision function value for each cluster.
        """
        # To be completed
        # Do not forget that this method is called in predict_preference (line 42) and therefor should return well-organized data for it to work.
        U = [self.models[k].predict_utility(X) for k in range(self.K)]
        U = np.concatenate(U, axis=1)
        return U
