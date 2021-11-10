nextflow.enable.dsl = 2

params.out = '.'
params.splits = 20

process read_data {

    tag "${phenotypes[PHENO]}"

    input:
        file DATA from expression
        tuple val(PHENO), val(WHICH_CONTROLS),val(WHICH_GROUP) from experiments
        file GXG from string

    output:
        set val("pheno=${phenotypes[PHENO]};controls=${phenotypes[WHICH_CONTROLS]};subgroup=${phenotypes[WHICH_GROUP]}"), "Xy.npz", "A.npz"

    script:
        template 'data/makeXyA.py'

}

process split_data {

    input:
        set PARAMS, file(DATA), file(NET)
        each I from 0..(params.splits - 1)
        val SPLITS from params.splits

    output:
        set val("${PARAMS};fold=${I}"), "Xy_train.npz", "Xy_test.npz", file(NET) into splits

    script:
    template 'data_processing/train_test_split.py'

}

splits.into { splits_logreg; splits_galore; splits_logistic_graph_lasso; splits_scones; splits_gbdt; splits_stg }

lambdas_logreg = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

process logreg {

    tag "${PARAMS};lambda=${LAMBDA}"

    input:
        set PARAMS, file(TRAIN), file(TEST), file(NET) from splits_logreg
        each LAMBDA from lambdas_logreg

    output:
        set val("model=logreg;${PARAMS};lambda=${LAMBDA}"), file(TEST), 'y_proba.npy' into predictions_logreg

    script:
    template 'classifier/logreg.py'

}

lambdas_1_galore = [0.05, 0.1, 0.15, 0.2, 0.25]
lambdas_2_galore = [0.05, 0.1, 0.15, 0.2, 0.25]

process galore {

    tag "${PARAMS};lambda_1=${LAMBDA_1};lambda_2=${LAMBDA_2}"
    errorStrategy 'ignore'

    input:
        set PARAMS, file(TRAIN), file(TEST), file(NET) from splits_galore
        each LAMBDA_1 from lambdas_1_galore
        each LAMBDA_2 from lambdas_2_galore

    output:
        set val("model=galore;${PARAMS};lambda_1=${LAMBDA_1};lambda_2=${LAMBDA_2}"), file(TEST), 'y_proba.npy' into predictions_galore

    script:
    template 'classifier/galore.py'

}

process logistic_graph_lasso {

    tag "${PARAMS};lambda_1=${LAMBDA_1};lambda_2=${LAMBDA_2}"
    errorStrategy 'ignore'

    input:
        set PARAMS, file(TRAIN), file(TEST), file(NET) from splits_logistic_graph_lasso
        each LAMBDA_1 from lambdas_1_galore
        each LAMBDA_2 from lambdas_2_galore

    output:
        set val("model=logistic_graph_lasso;${PARAMS};lambda_1=${LAMBDA_1};lambda_2=${LAMBDA_2}"), file(TEST), 'y_proba.npy' into predictions_logistic_graph_lasso

    script:
    template 'classifier/logistic_graph_lasso.py'

}

numleaves_gbdt = [20, 40, 60, 80, 100]
alphas_gbdt = [0.001, 0.01, 0.1]
lambdas_gbdt = [0.001, 0.01, 0.1]

process gbdt {

    tag "${PARAMS};num_leaves=${NUM_LEAVES};alpha=${ALPHA};lambda=${LAMBDA}"

    input:
        set PARAMS, file(TRAIN), file(TEST), file(NET) from splits_gbdt
        each NUM_LEAVES from numleaves_gbdt
        each ALPHA from alphas_gbdt
        each LAMBDA from lambdas_gbdt
    output:
        set val("model=gbdt;${PARAMS};num_leaves=${NUM_LEAVES};alpha=${ALPHA};lambda=${LAMBDA}"), file(TEST), 'y_proba.npy' into predictions_gbdt

    script:
    template 'classifier/gbdt.py'

}

learning_rate_stg = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
sigmas_stg = [1e-1, 3e-1, 5e-1, 7e-1, 1]
lambdas_stg = [5e-5, 5e-4, 5e-3, 5e-2, 5e-1]

process stg {

    tag "${PARAMS};learning_rate=${LR};sigma=${SIGMA};lambda=${LAMBDA}"

    input:
        set PARAMS, file(TRAIN), file(TEST), file(NET) from splits_stg
        each LR from learning_rate_stg
        each SIGMA from sigmas_stg
        each LAMBDA from lambdas_stg
    output:
        set val("model=stg;${PARAMS};learning_rate=${LR};sigma=${SIGMA};lambda=${LAMBDA}"), file(TEST), 'y_proba.npy' into predictions_stg

    script:
    template 'classifier/stg.py'

}

etas_scones = [0, 0.01, 0.05, 0.1, 0.2, 0.3]
lambdas_scones = [0, 0.5, 0.75, 1, 3, 5, 7, 9, 10]

process scones {

    tag "${PARAMS};eta=${ETA};lambda=${LAMBDA}"

    input:
        set PARAMS, file(TRAIN), file(TEST), file(NET) from splits_scones
        file STRING from string
        each ETA from etas_scones
        each LAMBDA from lambdas_scones

    output:
        set val("model=scones;${PARAMS};eta=${ETA};lambda=${LAMBDA}"), file(TRAIN), file(TEST), 'selected.scones.tsv' into features_scones

    script:
    template 'feature_selection/scones_cv.R'

}

features_scones.set { features }

process svm {

    tag "${PARAMS}"
    errorStrategy 'ignore'
    validExitStatus 0,77

    input:
        set PARAMS, file(TRAIN), file(TEST), file(SELECTED_FEATURES) from features

    output:
        set PARAMS, file(TEST), 'y_proba.npy' into predictions_svm

    script:
    template 'classifier/svc.py'

}

predictions_svm
    .mix(predictions_galore, predictions_logistic_graph_lasso, predictions_logreg, predictions_gbdt, predictions_stg)
    .set { predictions }

process analyze_predictions {

    tag "${PARAMS}"

    input:
        set PARAMS, file(TEST), file(Y_PROBA) from predictions

    output:
        file 'prediction_stats' into prediction_analysis

    script:
    template 'analysis/roc.py'

}

workflow benchmark {
    main:
        read_data()
        split_data()
        analyze_predictions()
    emit:
        analyze_predictions.out.collectFile(skip: 1, keepHeader: true)
}
