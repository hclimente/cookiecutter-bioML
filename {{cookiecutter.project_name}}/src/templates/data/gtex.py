def download_gtex_data(gtex_data_file):
    """Downloads the provided GTEx v8 table.

    Args:
        gtex_data_file (str): The GTEx file to download.

    Returns:
        None
    """
    import urllib.request

    gtex_data_path = "https://storage.googleapis.com/gtex_analysis_v8/rna_seq_data/"

    # Download the GTEx data
    print("Downloading {} from GTEx...".format(gtex_data_file))
    urllib.request.urlretrieve(gtex_data_path + gtex_data_file, "./" + gtex_data_file)
    print("Done!")


def get_gtex_gene_expression():
    download_gtex_data(
        "GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz"
    )


if __name__ == "__main__":
    get_gtex_gene_expression()
