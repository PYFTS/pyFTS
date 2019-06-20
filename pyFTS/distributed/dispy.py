import dispy as dispy, dispy.httpd, logging
from pyFTS.common import Util
import numpy as np


def start_dispy_cluster(method, nodes):
    """
    Start a new Dispy cluster on 'nodes' to execute the method 'method'

    :param method: function to be executed on each cluster node
    :param nodes: list of node names or IP's.
    :return: the dispy cluster instance and the http_server for monitoring
    """

    cluster = dispy.JobCluster(method, nodes=nodes, loglevel=logging.DEBUG, ping_interval=1000)

    http_server = dispy.httpd.DispyHTTPServer(cluster)

    return cluster, http_server


def stop_dispy_cluster(cluster, http_server):
    """
    Stop a dispy cluster and http_server

    :param cluster:
    :param http_server:
    :return:
    """
    cluster.wait()  # wait for all jobs to finish

    cluster.print_status()

    http_server.shutdown()  # this waits until browser gets all updates
    cluster.close()


def get_number_of_cpus(cluster):
    cpus = 0
    for dispy_node in cluster.status().nodes:
        cpus += dispy_node.cpus

    return cpus


def simple_model_train(model, data, parameters):
    """
    Cluster function that receives a FTS instance 'model' and train using the 'data' and 'parameters'

    :param model: a FTS instance
    :param data: training dataset
    :param parameters: parameters for the training process
    :return: the trained model
    """
    model.train(data, **parameters)
    return model


def distributed_train(model, train_method, nodes, fts_method, data, num_batches=10,
                      train_parameters={}, **kwargs):
    import dispy, dispy.httpd, datetime

    batch_save = kwargs.get('batch_save', False)  # save model between batches

    batch_save_interval = kwargs.get('batch_save_interval', 1)

    file_path = kwargs.get('file_path', None)

    cluster, http_server = start_dispy_cluster(train_method, nodes)

    print("[{0: %H:%M:%S}] Distrituted Train Started with {1} CPU's"
          .format(datetime.datetime.now(), get_number_of_cpus(cluster)))

    jobs = []
    n = len(data)
    batch_size = int(n / num_batches)
    bcount = 1
    for ct in range(model.order, n, batch_size):
        if model.is_multivariate:
            ndata = data.iloc[ct - model.order:ct + batch_size]
        else:
            ndata = data[ct - model.order: ct + batch_size]

        tmp_model = fts_method()

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
                Util.persist_obj(model, file_path)

        else:
            print(job.exception)
            print(job.stdout)

        print("[{0: %H:%M:%S}] Finished batch ".format(datetime.datetime.now()) + str(job.id))

    print("[{0: %H:%M:%S}] Distrituted Train Finished".format(datetime.datetime.now()))

    stop_dispy_cluster(cluster, http_server)

    return model



def simple_model_predict(model, data, parameters):
    return model.predict(data, **parameters)



def distributed_predict(model, parameters, nodes, data, num_batches):
    import dispy, dispy.httpd

    cluster, http_server = start_dispy_cluster(simple_model_predict, nodes)

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

    stop_dispy_cluster(cluster, http_server)

    return ret
