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
model_algorithms = ['logreg', 'random_forest', 'svc', 'knn']

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
        tuple val("model=${MODEL};${PARAMS}"), path(TEST_NPZ), path('y_proba.npz')

    script:
        template "${mode}/${MODEL}.py"

}

process analyze_predictions {

    tag "${PARAMS}"

    input:
        tuple val(PARAMS), path(TEST_NPZ), path(Y_PROBA)

    output:
        path 'prediction_stats'

    script:
        template 'analysis/roc.py'

}


workflow models {
    take: data
    main:
        split_data(data, 0..(params.splits - 1), params.splits)
        feature_selection(feature_selection_algorithms, split_data.out, config)
        model(model_algorithms, feature_selection.out, config)
        analyze_predictions(model.out)
    emit:
        analyze_predictions.out.collectFile(name: "${params.out}/sample.txt", skip: 1, keepHeader: true)
}

workflow {
    main:
        simulate_data(100, 20)
        models(simulate_data.out)
}
