nextflow.enable.dsl = 2

params.out = '.'
params.splits = 5

process read_data {

    tag "${phenotypes[PHENO]}"

    input:
        file DATA from expression
        tuple val(PHENO), val(WHICH_CONTROLS),val(WHICH_GROUP) from experiments
        file GXG from string

    output:
        tuple val("pheno=${phenotypes[PHENO]};controls=${phenotypes[WHICH_CONTROLS]};subgroup=${phenotypes[WHICH_GROUP]}"), "Xy.npz", "A.npz"

    script:
        template 'data/makeXyA.py'

}

process simulate_data {
    
    input:
        val NUM_SAMPLES
        val NUM_FEATURES

    output:
        tuple val("test"), path("simulation.npz")

    script:
        template "simulation/linear_0.py"

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

lambdas_logreg = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

process logreg {

    tag "${PARAMS};lambda=${LAMBDA}"

    input:
        tuple val(PARAMS), path(TRAIN_NPZ), path(TEST_NPZ)
        path PARAMS_FILE

    output:
        tuple val("model=logreg;${PARAMS}"), path(TEST_NPZ), path('y_proba.npz')

    script:
        template 'classifiers/logreg.py'

}

process random_forest {

    input:
        tuple val(PARAMS), path(TRAIN_NPZ), path(TEST_NPZ)
        path SELECTED
        path PARAMS_FILE

    output:
        tuple val("model=random_forest;${PARAMS}"), path(TEST_NPZ), path('y_proba.npz')

    script:
        template "classifiers/random_forest.py"

}

process analyze_predictions {

    tag "${PARAMS}"

    input:
        tuple val(PARAMS), path(TEST_NPZ), path(Y_PROBA)

    output:
        file 'prediction_stats'

    script:
        template 'analysis/roc.py'

}

json_logreg = file("$baseDir/src/templates/classifier/logreg.json")
json_rf = file("$baseDir/src/templates/classifier/random_forest.json")

workflow {
    main:
        simulate_data(100, 20)
        split_data(simulate_data.out, 0..(params.splits - 1), params.splits)
        logreg(split_data.out, json_logreg)
        random_forest(split_data.out, ".", json_rf)
        analyze_predictions(logreg.out)
    emit:
        analyze_predictions.out.collectFile(skip: 1, keepHeader: true)
}
