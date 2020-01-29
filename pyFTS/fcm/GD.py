import  numpy as np


def GD(data, model, **kwargs):
    alpha = kwargs.get('alpha', 0.1)
    momentum = kwargs.get('momentum', None)
    iterations = kwargs.get('iterations', 1)
    num_concepts = model.partitioner.partitions
    weights = [np.random.normal(0,.1,(num_concepts,num_concepts)) for k in range(model.order)]
    bias = [np.random.normal(0,.1,num_concepts) for k in range(model.order)]
    last_gradientW = [None for k in range(model.order) ]
    last_gradientB = [None for k in range(model.order)]
    for it in np.arange(iterations):
        for i in np.arange(model.order, len(data)):
            #i = np.random.randint(model.order, len(data)-model.order)
            sample = data[i-model.order : i]
            target = model.partitioner.fuzzyfy(data[i], mode='vector')
            #target = data[i]
            model.fcm.weights = weights
            model.fcm.bias = bias
            inputs = model.partitioner.fuzzyfy(sample, mode='vector')
            #activations = model.fcm.activate(inputs)
            activations = [model.fcm.activation_function(inputs[k]) for k in np.arange(model.order)]
            forecast = model.predict(sample)[0]
            error = target - model.partitioner.fuzzyfy(forecast, mode='vector') #)**2
            #error = target - forecast
            #if str(error) == 'nan' or error == np.nan or error == np.Inf:
            #print(error)
            print(np.dot(error,error))

            for k in np.arange(model.order):
                deriv = error * model.fcm.activation_function(activations[k], deriv=True)
                #deriv = error * activations[k]
                if momentum is not None:
                    if last_gradientW[k] is None:
                        last_gradientW[k] = deriv * inputs[k]
                        last_gradientB[k] = deriv

                    tmp_gradw = (momentum * last_gradientW[k]) + alpha*deriv*inputs[k]
                    weights[k] -= tmp_gradw
                    last_gradientW[k] = tmp_gradw

                    tmp_gradB = (momentum * last_gradientB[k]) + alpha * deriv
                    bias[k] -= tmp_gradB
                    last_gradientB[k] = tmp_gradB

                else:
                    weights[k] -= alpha*deriv*inputs[k]
                    bias[k] -= alpha*deriv

    return weights