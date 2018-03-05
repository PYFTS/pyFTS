import time
import matplotlib.pyplot as plt
import dill
import numpy as np


current_milli_time = lambda: int(round(time.time() * 1000))


def uniquefilename(name):
    if '.' in name:
        tmp = name.split('.')
        return  tmp[0] + str(current_milli_time()) + '.' + tmp[1]
    else:
        return name + str(current_milli_time())


def show_and_save_image(fig, file, flag, lgd=None):
    """
    Show and image and save on file
    :param fig: Matplotlib Figure object
    :param file: filename to save the picture
    :param flag: if True the image will be saved
    :param lgd: legend
    """
    if flag:
        plt.show()
        if lgd is not None:
            fig.savefig(file, additional_artists=lgd,bbox_inches='tight')  #bbox_extra_artists=(lgd,), )
        else:
            fig.savefig(file)
        plt.close(fig)


def enumerate2(xs, start=0, step=1):
    for x in xs:
        yield (start, x)
        start += step


def sliding_window(data, windowsize, train=0.8, inc=0.1):
    """
    Sliding window method of cross validation for time series
    :param data: the entire dataset
    :param windowsize: window size
    :param train: percentual of the window size will be used for training the models
    :param inc: percentual of data used for slide the window
    :return: window count, training set, test set
    """
    l = len(data)
    ttrain = int(round(windowsize * train, 0))
    ic = int(round(windowsize * inc, 0))
    for count in np.arange(0,l-windowsize+ic,ic):
        if count + windowsize > l:
            _end = l
        else:
            _end = count + windowsize
        yield (count,  data[count : count + ttrain], data[count + ttrain : _end]  )


def persist_obj(obj, file):
    """
    Persist an object on filesystem. This function depends on Dill package
    :param obj: object on memory
    :param file: file name to store the object
    """
    with open(file, 'wb') as _file:
        dill.dump(obj, _file)


def load_obj(file):
    """
    Load to memory an object stored filesystem. This function depends on Dill package
    :param file: file name where the object is stored
    :return: object
    """
    with open(file, 'rb') as _file:
        obj = dill.load(_file)
    return obj


def persist_env(file):
    """
    Persist an entire environment on file. This function depends on Dill package
    :param file: file name to store the environment
    """
    dill.dump_session(file)


def load_env(file):
    dill.load_session(file)


def simple_model_train(model, data, parameters):
    model.train(data, **parameters)
    return model


def distributed_train(model, train_method, nodes, fts_method, data, num_batches=10,
                      train_parameters={}, **kwargs):
    import dispy, dispy.httpd, datetime

    batch_save = kwargs.get('batch_save', False)  # save model between batches

    batch_save_interval = kwargs.get('batch_save_interval', 1)

    file_path = kwargs.get('file_path', None)

    cluster = dispy.JobCluster(train_method, nodes=nodes)  # , depends=dependencies)

    http_server = dispy.httpd.DispyHTTPServer(cluster)

    print("[{0: %H:%M:%S}] Distrituted Train Started".format(datetime.datetime.now()))

    jobs = []
    n = len(data)
    batch_size = int(n / num_batches)
    bcount = 1
    for ct in range(model.order, n, batch_size):
        if model.is_multivariate:
            ndata = data.iloc[ct - model.order:ct + batch_size]
        else:
            ndata = data[ct - model.order: ct + batch_size]

        tmp_model = fts_method(str(bcount))

        tmp_model.clone_parameters(model)

        job = cluster.submit(tmp_model, ndata, train_parameters)
        job.id = bcount  # associate an ID to identify jobs (if needed later)
        jobs.append(job)

        bcount += 1

    for job in jobs:
        print("[{0: %H:%M:%S}] Processing batch ".format(datetime.datetime.now()) + str(job.id))
        tmp = job()
        if job.status == dispy.DispyJob.Finished and tmp is not None:
            model.merge(tmp)

            if batch_save and (job.id % batch_save_interval) == 0:
                persist_obj(model, file_path)

        else:
            print(job.exception)
            print(job.stdout)

        print("[{0: %H:%M:%S}] Finished batch ".format(datetime.datetime.now()) + str(job.id))

    print("[{0: %H:%M:%S}] Distrituted Train Finished".format(datetime.datetime.now()))

    cluster.wait()  # wait for all jobs to finish

    cluster.print_status()

    http_server.shutdown()  # this waits until browser gets all updates
    cluster.close()

    return model


def simple_model_predict(model, data, parameters):
    return model.predict(data, **parameters)


def distributed_predict(model, parameters, nodes, data, num_batches):
    import dispy, dispy.httpd

    cluster = dispy.JobCluster(simple_model_predict, nodes=nodes)  # , depends=dependencies)

    http_server = dispy.httpd.DispyHTTPServer(cluster)

    jobs = []
    n = len(data)
    batch_size = int(n / num_batches)
    bcount = 1
    for ct in range(model.order, n, batch_size):
        if model.is_multivariate:
            ndata = data.iloc[ct - model.order:ct + batch_size]
        else:
            ndata = data[ct - model.order: ct + batch_size]

        job = cluster.submit(model, ndata, parameters)
        job.id = bcount  # associate an ID to identify jobs (if needed later)
        jobs.append(job)

        bcount += 1

    ret = []

    for job in jobs:
        tmp = job()
        if job.status == dispy.DispyJob.Finished and tmp is not None:
            if job.id < batch_size:
                ret.extend(tmp[:-1])
            else:
                ret.extend(tmp)
        else:
            print(job.exception)
            print(job.stdout)

    cluster.wait()  # wait for all jobs to finish

    cluster.print_status()

    http_server.shutdown()  # this waits until browser gets all updates
    cluster.close()

    return ret
