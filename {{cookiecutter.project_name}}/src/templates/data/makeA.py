#!/usr/bin/env python
import networkx as nx
import numpy as np
import pandas as pd


def makeA(gxg_file, X, genes):

    # sanitize genes
    genes, idx, counts = np.unique(genes, return_index=True, return_counts=True)
    genes = genes[counts == 1]
    idx = idx[counts == 1]
    X = X[:, idx]

    # read network
    gxg = pd.read_csv(gxg_file, sep="\\t")

    # remove genes not in the experiment
    gxg = gxg.loc[gxg.symbol1.isin(genes)]
    gxg = gxg.loc[gxg.symbol2.isin(genes)]

    # remove genes not in the network
    gxg_genes = set(gxg.symbol1) or set(gxg.symbol2)
    which = [g in gxg_genes for g in genes]
    genes = genes[which]
    X = X[:, which]

    G = nx.from_pandas_edgelist(gxg, "symbol1", "symbol2")
    A = nx.to_scipy_sparse_matrix(G, nodelist=genes)
    A.setdiag(1)

    return A, X, genes
