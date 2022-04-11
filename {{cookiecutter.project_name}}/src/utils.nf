import groovy.json.JsonBuilder
import groovy.json.StringEscapeUtils

def make_grid(models) {

    grid = []

    for (model in models) {

        keys = model.fixed_parameters.collect{key, value -> key}
        parameters = model.fixed_parameters.collect{key, value -> value}

        for (combination in parameters.combinations()) {

            fixed_parameters = [keys, combination]
                .transpose()
                .flatten()
                .collate(2, false)
                .collectEntries()
            parameters = [fixed_parameters: fixed_parameters,
                          cv_parameters: model.cv_parameters]

            builder = new JsonBuilder()
            builder(parameters)
            parameters = builder.toString()
            parameters = StringEscapeUtils.escapeJava(parameters)

            grid += [new Tuple(model.name, parameters)]

        }
    }

    return grid

}
