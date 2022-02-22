from enum import unique
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import random
from Cluster import MyKMeans
from sklearn.cluster import KMeans

SIZE = 20
LINEWIDTH = 3
FIGSIZE = (15, 12)

class Preprocess:
    """
    The class for data processing
    """

    @staticmethod
    def get_missing_values(data):
        """
        Returns columns with empty values.
            Parameters:
            -----------
                data: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
            Return:
            -------
                None
        """
        missing_values_columns = []
        missing_values = data.isnull().sum()
        for index, value in missing_values.items():
            if value != 0:
                missing_values_columns.append(index)
        return missing_values_columns

    @staticmethod
    def fill_missing_values(data, eps=0.7):
        """
        Fills nulls with the average of the columns if the number of nulls is greater than the specified threshold.
            Parameters:
            -----------
                data: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
                eps: float
                    the threshold
            Return:
            -------
                None
        """
        
        for column in Preprocess.get_missing_values(data):
            if data[column].isna().sum() / data.shape[0] > eps:
                data.drop(labels=column, axis=1, inplace=True)
            else:
                data[column].fillna(value=data[column].mean(), inplace=True)
        
    @staticmethod
    def get_cat_var(data):
        """
        Gets the list of the categorical variables.
            Parameters:
            -----------
                data: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
            Return:
            -------
                cat_features: list
                    the list of the categorical variables
        """
        cat_features = []
        for i in data.columns:
            if data.dtypes[i]=='object':
                cat_features.append(i)
        return cat_features

    @staticmethod
    def cast_cat_var(data):
        """
        Casts the the categorical variables to the numerical values.
        This example considers only a simple case for binary categorical variables.
            Parameters:
            -----------
                data: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
            Return:
            -------
                None
        """
        unique_values = data[Preprocess.get_cat_var(data)[0]].unique()
        data[Preprocess.get_cat_var(data)[0]] = data[Preprocess.get_cat_var(data)[0]].map({unique_values[0]:0,unique_values[1]:1})

    @staticmethod
    def get_label(data, label='diagnosis'):
        """
        Returns the target variable.
            Parameters:
            -----------
                data: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
            Return:
            -------
                y_true: numpy.ndarray
        """
        y_true = np.array(data[label])
        return y_true
    
    
    @staticmethod
    def get_features(data, label=None):
        """
        Returns the matrix of features.
            Parameters:
            -----------
                data: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
            Return:
            -------
                X: numpy.ndarray
        """

        if label != None:
            X = data.drop([label], axis=1)
        else: X = data
        return X

    @staticmethod
    def scale(features, method='StandardScaler'):
        """
        Scales the features.
            Parameters:
            -----------
                features: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
                method: str
                    the scaling method
            Return:
            -------
                X: numpy.ndarray
                   scaled features
        """

        if method == 'StandardScaler':
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        elif method == 'MinMaxScaler':
            scaler = MinMaxScaler()
            features = scaler.fit_transform(features)

        return features
    
    @staticmethod
    def drop_columns(data, columns=['id']):
        """
        Drops columns.
            Parameters:
            -----------
                data: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
                columns: list
                    the columns to be deleted
            Return:
            -------
                None
        """
        data.drop(labels=columns, axis=1, inplace=True)

    @staticmethod
    def determine_n_components(features, eps=0.999):
        """
        Determines the number of principal components to be used.
            Parameters:
            -----------
                features: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
                eps: float
                    threshold that determines the sufficient value of explained variance
            Return:
            -------
                number: int
                    the number of the principal components
                exp_var: numpy.ndarray
                    the array of the explained variance
        """
        pca = PCA().fit(features)
        exp_var = np.cumsum(pca.explained_variance_ratio_)

        number = 0
        for i in exp_var:
            number += 1
            if i >=eps:
                break

        return (number, exp_var)


class Vizualization():
    """
    A class that defines the methods needed for visualization of the results.
    """

    @staticmethod
    def plot_pca(features):
        """
        Builds a graph of dependence of the explained variance on the number of components.
            Parameters:
            -----------
                features: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
            Return:
            -------
                None
        """

        n = Preprocess.determine_n_components(features)[0]
        cum_var = Preprocess.determine_n_components(features)[1]

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.tick_params(axis="both", which="major", labelsize=SIZE)
        ax.plot(cum_var, color="k", linewidth=LINEWIDTH)
        ax.axvline(n, color="r", linestyle="dashed", linewidth=LINEWIDTH)
        ax.set_xlabel("Number of components", fontsize=SIZE)
        ax.set_ylabel("Cumulative explained variance", fontsize=SIZE)
        ax.text(x=n/2, y=cum_var.max()/2,
                s=f"A Number of components is {n}", size=SIZE, ha='center', va='center')
        ax.grid()

    @staticmethod
    def _associate_colors(y, model):
        """
        Associates colors for each clusters.
            Parameters:
            -----------
                model:
                    the model used for clustering
                y: numpy.ndarray
                    the labels of points
            Return:
            -------
                color_list: list
                    the list of colors where each position corresponds to the position of a point in the original feature matrix
        """

        color_list = list(y)

        for cl_idx in range(model.n_clusters):
            
            random.seed(42 + cl_idx)
            r, g, b = random.randint(0, 255)/255.0, random.randint(0, 255)/255.0, random.randint(0, 255)/255.0

            for i in range(len(y)):
                if y[i] == cl_idx:
                    color_list[i] = [r, g, b]

        return color_list

    @staticmethod
    def plot_clusters(features,
                      model, 
                      true_labels=None, 
                      inverse_axes=False):
        """
        Builds a graph of clusters.
            Parameters:
            -----------
                features: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
                model:
                    the model used for clustering
                true_label: numpy.ndarray
                    the true labels
                inverse_axes: bool
                    if "True" then swaps the axes on the graph
            Return:
            -------
                None
        """

        pred_labels = model.labels_
        fig, axs = plt.subplots(2, 1, figsize=FIGSIZE)
        for ax in axs.flat:
            ax.tick_params(axis="both", which="major", labelsize=SIZE)
            ax.set_xlabel("First component", fontsize=SIZE)
            ax.set_ylabel("Second component", fontsize=SIZE)
            ax.label_outer()

        i, j = 0, 1
        if inverse_axes:
            i, j = 1, 0

        axs[0].scatter(features[:, i], features[:, j], edgecolors=Vizualization._associate_colors(
            pred_labels, model), facecolor='None', linewidth=0.8*LINEWIDTH)
        axs[0].set_title('Prediction', fontsize=SIZE)
        if true_labels.size != 0:
            axs[1].scatter(features[:, i], features[:, j], edgecolors=Vizualization._associate_colors(
                true_labels, model), facecolor='None', linewidth=0.8*LINEWIDTH)
            axs[1].set_title('Original', fontsize=SIZE)


class ClusterNumber():
    """
    The class of methods for determining the optimal number of clusters
    """

    @staticmethod
    def elbow_method(model, data, k_max):
        """
        Impementation of the Elbow method.
            Parameters:
            -----------
                data: MxN numpy.ndarray
                    where M is the number of objects, N is the number of features
                k_max: int
                    the maximal number of clusters
                model:
                    the model used for clustering     
            Return:
            -------
                None
        """
    
        wcv = list()
        K = range(1, k_max)
        for k in K:

            if isinstance(model, MyKMeans):
                model = MyKMeans(k, seed=42)
            elif isinstance(model, KMeans):
                model = KMeans(k, random_state=42)

            model.fit(data)
            wcv.append(model.inertia_)
            
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.tick_params(axis="both", which="major", labelsize=SIZE)
        ax.plot(K, wcv, linewidth=LINEWIDTH, color="k", marker='o')
        ax.set_xlabel('k', fontsize=SIZE)
        ax.set_ylabel('Within-cluster variation', fontsize=SIZE)
        ax.grid()
