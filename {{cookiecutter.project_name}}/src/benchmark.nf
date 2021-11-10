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

    output:
        tuple val("test"), path("Xy.npz")

    script:
    """
    #!/usr/bin/env python

    import numpy as np

    x = np.random.rand(20,10)

    n = x.shape[0]

    x1 = x[:, 0]
    x2 = 2*x[:, 1]
    x3 = 4*x[:, 2]
    x4 = 8*x[:, 3]

    eps = np.random.normal(loc=0.0, scale=1.0, size=n)

    y = x1 + x2 + x3 + x4 + eps
    y = np.where(y > 0, 1, -1)

    np.savez("Xy.npz", X=x, y=y)
    """

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
        template 'feature_selectors/lars.py'

}

process predict {

    input:
        tuple val(PARAMS), path(TRAIN_NPZ), path(TEST_NPZ), path(SELECTED)
        path PARAMS_FILE

    output:
        tuple val(PARAMS), path(TEST_NPZ), path("y_proba.npz")

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

json = file("lars.json")
json2 = file("rf.json")

workflow {
    main:
        simulate_data()
        split_data(simulate_data.out, 0..(params.splits - 1), params.splits)
        logreg(split_data.out, json)
        analyze_predictions(logreg.out)
    emit:
        analyze_predictions.out.collectFile(skip: 1, keepHeader: true)
}
