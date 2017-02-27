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
    if flag:
        plt.show()
        if lgd is not None:
            fig.savefig(uniquefilename(file), additional_artists=lgd,bbox_inches='tight')  #bbox_extra_artists=(lgd,), )
        else:
            fig.savefig(uniquefilename(file))
        plt.close(fig)


def enumerate2(xs, start=0, step=1):
    for x in xs:
        yield (start, x)
        start += step

def sliding_window(data, windowsize, train=0.8, inc=0.1):
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
    with open(file, 'wb') as _file:
        dill.dump(obj, _file)

def load_obj(file):
    with open(file, 'rb') as _file:
        obj = dill.load(_file)
    return obj

def persist_env(file):
    dill.dump_session(file)

def load_env(file):
    dill.load_session(file)