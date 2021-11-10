#!/usr/bin/env Rscript
'
Input variables:
    - GXG: path to the PPIN
Output files:
    - Xy.npz
    - A.npz
'

library(igraph)
library(martini)
library(reticulate)
library(tidyverse)

# read ppi
string <- read_tsv('${GXG}', col_types = 'cc') %>%
    graph_from_data_frame(directed = FALSE) %>%
    simplify %>%
    set_vertex_attr(., "gene", value = names(V(.)))

A <- as_adj(string)
diag(A) <- 1

# simulate causal subnetwork
genes <- colnames(A)

g <- sample(genes, 1)
seed <- martini:::subvert(string, "gene", g)[1]

causal <- neighbors(string,seed)\$gene %>% na.omit %>% unique
causal <- c(causal, g)

causal <- genes[genes %in% causal]

# simulate phenotypes
weight = rnorm(length(causal))
n_samples = 1000
prevalence = 0.1
n_condition = as.integer(n_samples * prevalence)

X <- matrix(rchisq(length(genes) * 1000, df = 3), ncol=1000)
X_causal <- X[genes %in% causal,]
Y <- weight %*% X_causal + rnorm(length(causal))

Y.sorted <- sort(Y, index.return = TRUE)
cases <- head(Y.sorted[['ix']], n = n_condition)
controls <- tail(Y.sorted[['ix']], n = n_condition)

Y[cases] <- 1
Y[controls] <- -1
Y[-c(cases,controls)] <- NA

y <- Y[!is.na(Y)]
X <- t(X[, !is.na(Y)])

# save data
scp <- import("scipy.sparse")
scp\$save_npz("A.npz", A)

np <- import("numpy")
np\$savez("Xy.npz", X=X, Y=y, genes=genes)
np\$savez("causal.npz", causal=causal, weights=weight)
