nextflow.enable.dsl = 2

// Parameters
/////////////////////////////////////////////////////////
params.out = '.'
params.splits = 5
params.mode = "classification"

config = file("${params.out}/config.yaml")
mode = params.mode

// TODO take the algorithms from the config file
feature_selection_algorithms = ['all_features']
model_algorithms = ['logistic_regression', 'random_forest', 'svc', 'knn']
performance_metrics = ['auc_roc', 'tpr_fpr']

process simulate_data {

    input:
        val NUM_SAMPLES
        val NUM_FEATURES

    output:
        tuple val("test"), path("simulation.npz")

    script:
        template "simulation/categorical_1.py"

}

process split_data {

    input:
        tuple val(PARAMS), path(DATA_NPZ)
        each I
        val SPLITS

    output:
        tuple val("${PARAMS};fold=${I}"), path("Xy_train.npz"), path("Xy_test.npz")

    script:
        template 'data/kfold.py'

}

process feature_selection {

    tag "${MODEL};${PARAMS}"

    input:
        each MODEL
        tuple val(PARAMS), path(TRAIN_NPZ), path(TEST_NPZ)
        path PARAMS_FILE

    output:
        tuple val("feature_selection=${MODEL}"), path(TRAIN_NPZ), path(TEST_NPZ), path('scores.npz')

    script:
        template "feature_selection/${MODEL}.py"

}

process model {

    tag "${MODEL};${PARAMS}"

    input:
        each MODEL
        tuple val(PARAMS), path(TRAIN_NPZ), path(TEST_NPZ), path(SCORES_NPZ)
        path PARAMS_FILE

    output:
        tuple val("model=${MODEL};${PARAMS}"), path(TEST_NPZ), path('y_proba.npz'), path('y_pred.npz')

    script:
        template "${mode}/${MODEL}.py"

}

process performance {

    tag "${METRIC};${PARAMS}"

    input:
        each METRIC
        tuple val(PARAMS), path(TEST_NPZ), path(PROBA_NPZ), path(PRED_NPZ)

    output:
        path 'performance.tsv'

    script:
        template "performance/${METRIC}.py"

}


workflow models {
    take: data
    main:
        split_data(data, 0..(params.splits - 1), params.splits)
        feature_selection(feature_selection_algorithms, split_data.out, config)
        model(model_algorithms, feature_selection.out, config)
        performance(performance_metrics, model.out)
    emit:
        performance.out.collectFile(name: "${params.out}/sample.txt", skip: 1, keepHeader: true)
}

workflow {
    main:
        simulate_data(100, 20)
        models(simulate_data.out)
}
