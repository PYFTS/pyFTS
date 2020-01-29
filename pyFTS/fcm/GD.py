import  numpy as np


def GD(data, model, **kwargs):
    alpha = kwargs.get('alpha', 0.1)
    momentum = kwargs.get('momentum', None)
    iterations = kwargs.get('iterations', 1)
    num_concepts = model.partitioner.partitions
    weights = [np.random.normal(0,.01,(num_concepts,num_concepts)) for k in range(model.order)]
    last_gradient = [None for k in range(model.order) ]
    for it in np.arange(iterations):
        for i in np.arange(model.order, len(data)):
            #i = np.random.randint(model.order, len(data)-model.order)
            sample = data[i-model.order : i]
            target = data[i]
            model.fcm.weights = weights
            inputs = model.partitioner.fuzzyfy(sample, mode='vector')
            activations = model.fcm.activate(inputs)
            forecast = model.predict(sample)[0]
            error = target - forecast #)**2
            if str(error) == 'nan' or error == np.nan or error == np.Inf:
                print('error')
            print(error)
            for k in np.arange(model.order):
                deriv = error * model.fcm.activation_function(activations[k], deriv=True)
                if momentum is not None:
                    if last_gradient[k] is None:
                        last_gradient[k] = deriv*inputs[k]

                    tmp_grad = (momentum * last_gradient[k]) + alpha*deriv*inputs[k]
                    weights[k] -= tmp_grad
                    last_gradient[k] = tmp_grad
                else:
                    weights[k] -= alpha*deriv*inputs[k]

    return weights