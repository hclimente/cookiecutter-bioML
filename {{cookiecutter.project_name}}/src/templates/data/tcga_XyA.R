#!/usr/bin/env Rscript
'
Input variables:
    - CANCER: TCGA Study Abbreviations (e.g. BRCA)
    - GXG: path to the PPIN
Output files:
    - Xy.npz
'
library(SummarizedExperiment)
library(TCGAbiolinks)
library(tidyverse)
library(reticulate)

query <- GDCquery(project = "TCGA-${PHENO}",
                  data.category = "Transcriptome Profiling",
                  data.type = "Gene Expression Quantification",
                  workflow.type = "HTSeq - FPKM-UQ")
GDCdownload(query)
data <- GDCprepare(query)

# select samples from tumor and normal tissue
sample_info <- as_tibble(colData(data))

sample_type <- colData(data)\$sample_type
accepted_sample_types <- c("Primary Tumor", "Solid Tissue Normal")
samples_keep <- sample_type %in% accepted_sample_types

# save data as numpy object
X <- t(assay(data)[,samples_keep])
y <- as.integer(sample_type[samples_keep] == "Primary Tumor") * 2 - 1
genes <- rowData(data)\$external_gene_name

np <- import("numpy")
scp <- import("scipy.sparse")
read_net <- import('templates.io.makeA')

if ("${GXG}" != '') {
    # read network
    net = read_net\$makeA("${GXG}", X, genes)
    A = net[[1]]
    X = net[[2]]
    genes = net[[3]]

    scp\$save_npz("A.npz", A)
}

np\$savez("Xy.npz", X=X, Y=y, genes=genes)
