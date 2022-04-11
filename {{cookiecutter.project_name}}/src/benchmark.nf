nextflow.enable.dsl = 2
config = file("${params.out}/config.yaml")

// Parameters
/////////////////////////////////////////////////////////
params.out = '.'
params.splits = 5
params.mode = "regression"

mode = params.mode

include { make_grid } from './utils.nf'

process simulate_data {

    tag "${SIMULATION}(${NUM_SAMPLES},${NUM_FEATURES})"
    afterScript 'mv scores.npz causal.npz'

    input:
        each SIMULATION
        each NUM_SAMPLES
        each NUM_FEATURES

    output:
        tuple val("${SIMULATION}(${NUM_SAMPLES},${NUM_FEATURES}"), path("simulation.npz"), path('causal.npz')

    script:
        template "simulation/${SIMULATION}.py"

}

process split_data {

    tag "${PARAMS},${I})"

    input:
        tuple val(PARAMS), path(DATA_NPZ), path(CAUSAL_NPZ)
        each I
        val SPLITS

    output:
        tuple val("${PARAMS},${I})"), path("Xy_train.npz"), path("Xy_test.npz"), path(CAUSAL_NPZ)

    script:
        template 'data/kfold.py'

}

process feature_selection {

    tag "${MODEL};${PARAMS}"
    afterScript 'mv scores.npz scores_feature_selection.npz'

    input:
        tuple val(PARAMS), path(TRAIN_NPZ), path(TEST_NPZ), path(CAUSAL_NPZ), val(MODEL), val(MODEL_PARAMS)
        path PARAMS_FILE

    output:
        tuple val("feature_selection=${MODEL}(${MODEL_PARAMS});${PARAMS}"), path(TRAIN_NPZ), path(TEST_NPZ), path(CAUSAL_NPZ), path('scores_feature_selection.npz')

    script:
        template "feature_selection/${MODEL}.py"

}

process prediction {

    tag "${MODEL};${PARAMS}"
    afterScript 'mv scores.npz scores_model.npz'
    errorStrategy { task.exitStatus == 77 ? 'ignore' : 'terminate' }

    input:
        tuple val(PARAMS), path(TRAIN_NPZ), path(TEST_NPZ), path(CAUSAL_NPZ), path(SCORES_NPZ), val(MODEL), val(MODEL_PARAMS)
        path PARAMS_FILE

    output:
        tuple val("model=${MODEL}(${MODEL_PARAMS});${PARAMS}"), path(TEST_NPZ), path(CAUSAL_NPZ), path(SCORES_NPZ), path('y_proba.npz'), path('y_pred.npz')

    script:
        template "${mode}/${MODEL}.py"

}

process performance {

    tag "${METRIC};${PARAMS}"

    input:
        each METRIC
        tuple val(PARAMS), path(TEST_NPZ), path(CAUSAL_NPZ), path(SCORES_NPZ), path(PROBA_NPZ), path(PRED_NPZ)

    output:
        path 'performance.tsv'

    script:
        template "performance/${METRIC}.py"

}

feature_selection_grid = make_grid(params.feature_selection)
prediction_grid = make_grid(params.prediction)

workflow models {
    take: data
    main:
        split_data(data, 0..(params.splits - 1), params.splits)
        feature_selection(split_data.out.combine(feature_selection_grid), config)
        prediction(feature_selection.out.combine(prediction_grid), config)
        performance(params.performance_metrics, prediction.out)
    emit:
        performance.out.collectFile(name: "${params.out}/performance.tsv", skip: 1, keepHeader: true)
}

workflow {
    main:
        simulate_data(params.simulation_models, params.num_samples, params.num_features)
        models(simulate_data.out)
}
