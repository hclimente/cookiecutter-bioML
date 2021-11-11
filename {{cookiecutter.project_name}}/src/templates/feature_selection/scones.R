#!/usr/bin/env Rscript
'
Input variables:
    - TRAIN: path of a numpy array with x.
    - PPI: TSV with protein-protein interaction
Output files:
    - selected.npy
'

library(igraph)
library(martini)
library(reticulate)
library(tidyverse)

options(echo=TRUE) # if you want see commands in output file
args <- commandArgs(trailingOnly = TRUE)
string <- "${PPI}"

np <- import("numpy")

i <- np\$load("${TRAIN}", allow_pickle=TRUE)

genes = i\$f[['genes']]
X = i\$f[['X']]
colnames(X) <- genes
Y = i\$f[['Y']]

gxg <- read_tsv(string)
net <- graph_from_data_frame(gxg, directed=FALSE)
net <- set_edge_attr(net, "weight", value=1)

res <- scones.cv_(X, Y, genes, net)

tibble(gene = names(V(res))) %>%
  # compute R2 i.e. SConES' score
  mutate(r2 = apply(X[, gene], 2, cor, Y, use="pairwise.complete.obs"),
         r2 = ifelse(is.na(r2), 0.01, r2)) %>%
  write_tsv('selected.scones.tsv')
