#!/usr/bin/env Rscript
library(igraph)
library(martini)
library(reticulate)
library(tidyverse)

np <- import("numpy")

i <- np\$load("${TRAIN}", allow_pickle=TRUE)

X = i\$f[['X']]
Y = i\$f[['Y']]
genes = i\$f[['genes']]

gxg <- read_tsv("${STRING}")
net <- graph_from_data_frame(gxg, directed=FALSE)
net <- set_edge_attr(net, "weight", value=1)

res <- scones_(X, Y, genes, net, eta=${ETA}, lambda=${LAMBDA})

tibble(gene = names(V(res))) %>%
    write_tsv('selected.scones.tsv')
