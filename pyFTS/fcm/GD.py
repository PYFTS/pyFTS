import  numpy as np


def GD(data, model, alpha, momentum=0.5):
    num_concepts = model.partitioner.partitions
    weights=[np.random.normal(0,.01,(num_concepts,num_concepts)) for k in range(model.order)]
    last_gradient = [None for k in range(model.order) ]
    for i in np.arange(model.order, len(data)):
        sample = data[i-model.order : i]
        target = data[i]
        model.fcm.weights = weights
        inputs = model.partitioner.fuzzyfy(sample, mode='vector')
        activations = [model.fcm.activation_function(inputs[k]) for k in np.arange(model.order)]
        forecast = model.predict(sample)[0]
        error = target - forecast #)**2
        if error == np.nan:
            pass
        print(error)
        for k in np.arange(model.order):
            deriv = error * model.fcm.activation_function(activations[k], deriv=True)
            if momentum is not None:
                if last_gradient[k] is None:
                    last_gradient[k] = deriv*inputs[k]

                tmp_grad = (momentum * last_gradient[k]) + alpha*deriv*inputs[k]
                weights[k] -= tmp_grad
            else:
                weights[k] -= alpha*deriv*inputs[k]

    return weights