nextflow.enable.dsl = 2

// Parameters
/////////////////////////////////////////////////////////
params.out = '.'
params.splits = 5
params.mode = "classification"

config = file("${params.out}/config.yaml")
mode = params.mode

num_samples = [20]
num_features = [20]

simulation_models = ['categorical_1']
feature_selection_algorithms = ['all_features']
model_algorithms = ['logistic_regression', 'random_forest', 'svc', 'knn']
performance_metrics = ['auc_roc', 'tpr_fpr']

process simulate_data {

    tag "${SIMULATION}(${NUM_SAMPLES},${NUM_FEATURES})"

    input:
        each SIMULATION
        each NUM_SAMPLES
        each NUM_FEATURES

    output:
        tuple val("${SIMULATION}(${NUM_SAMPLES},${NUM_FEATURES}"), path("simulation.npz")

    script:
        template "simulation/${SIMULATION}.py"

}

process split_data {

    tag "${PARAMS},${I})"

    input:
        tuple val(PARAMS), path(DATA_NPZ)
        each I
        val SPLITS

    output:
        tuple val("${PARAMS},${I})"), path("Xy_train.npz"), path("Xy_test.npz")

    script:
        template 'data/kfold.py'

}

process feature_selection {

    tag "${MODEL};${PARAMS}"
    afterScript 'mv scores.npz scores_feature_selection.npz'

    input:
        each MODEL
        tuple val(PARAMS), path(TRAIN_NPZ), path(TEST_NPZ)
        path PARAMS_FILE

    output:
        tuple val("feature_selection=${MODEL};${PARAMS}"), path(TRAIN_NPZ), path(TEST_NPZ), path('scores_feature_selection.npz')

    script:
        template "feature_selection/${MODEL}.py"

}

process model {

    tag "${MODEL};${PARAMS}"
    afterScript 'mv scores.npz scores_model.npz'

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
        performance.out.collectFile(name: "${params.out}/performance.tsv", skip: 1, keepHeader: true)
}

workflow {
    main:
        simulate_data(simulation_models, 100, 20)
        models(simulate_data.out)
}
