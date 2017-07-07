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


def showAndSaveImage(fig,file,flag,lgd=None):
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