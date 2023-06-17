import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from mrmr import mrmr_classif

import matplotlib.pyplot as plt
import seaborn as sns


def dataset(file_path="pokemon.csv"):
    df = pd.read_csv(file_path)
    df = df[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed", "Total"]]
    X = df[["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]]
    y = df["Total"]
    return df, X, y


def spearman_correlation(df):
    r = df.corr(method="spearman")
    plt.figure(figsize=(8, 5))
    heatmap = sns.heatmap(
        r,
        vmin=-1,
        vmax=1,
        annot=True,
        cmap="vlag",
        alpha=0.8,
        linewidths=0.5,
        fmt=".2f",
        mask=np.triu(np.ones_like(r, dtype=bool), 1),
        cbar_kws={"shrink": 1, "orientation": "horizontal"},
    )
    plt.tick_params(axis="both", bottom=False, left=False)
    plt.title("Spearman's Rank Correlation Coefficient\n", fontsize=12, color="#3d3d3d")
    plt.axis('off')
    plt.savefig('spearman_correlation.png', bbox_inches='tight', pad_inches = 0)
    return r


def pca_plot(X):
    pca = PCA()
    pca.fit(X.T)
    per_var = 100 * pca.explained_variance_ratio_

    plt.figure(figsize=(8, 5))
    plt.plot([i for i in range(1, 7)], per_var)
    plt.ylabel("Fraction of Variance Explained", fontsize=12)
    plt.xlabel("Principal Component", fontsize=12)
    plt.title("Scree Plot", fontsize=12)
    plt.savefig('pca_plot.png', bbox_inches='tight', pad_inches = 0)
    plt.show()
    return np.cumsum(per_var)


def mRMR(X, y, k):
    selected_features = mrmr_classif(X=X, y=y, K=k)
    return selected_features


def lasso_plot(X, y, lmbda=900):
    lasso = Lasso(alpha=lmbda, tol=0.1)
    lasso.fit(X, y)
    lasso_beta = lasso.coef_

    data = {"Features": X.columns, "Coefs": lasso_beta}
    df = pd.DataFrame(data=data).sort_values(by=["Coefs"], ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=df["Coefs"], y=df["Features"], color="#4D81B6")
    sns.despine(left=True, bottom=True)
    plt.savefig('lasso_plot.png', bbox_inches='tight', pad_inches = 0)
    plt.show()
    return list(df["Features"])


def rf_permutation(X, y):
    rf = RandomForestRegressor(n_estimators=30)
    rf.fit(X, y)

    p_importance = permutation_importance(rf, X, y, n_repeats=25, random_state=12)
    p_importances = p_importance.importances_mean
    perm_sort = p_importances.argsort()

    perm_imp = pd.DataFrame(
        {"Features": X.columns[perm_sort], "Importance": p_importances[perm_sort]}
    )
    perm_imp["Features"] = pd.Categorical(
        perm_imp["Features"], categories=perm_imp["Features"], ordered=True
    )
    perm_imp = perm_imp.sort_values(by=["Importance"], ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=list(perm_imp["Importance"]), y=list(perm_imp["Features"]), color="#AE6464"
    )
    sns.despine(left=True, bottom=True)
    plt.savefig('rf_permutation.png', bbox_inches='tight', pad_inches = 0)
    plt.show()
    return list(perm_imp["Features"])
