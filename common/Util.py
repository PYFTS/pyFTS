import time
import matplotlib.pyplot as plt


current_milli_time = lambda: int(round(time.time() * 1000))


def uniquefilename(name):
    if '.' in name:
        tmp = name.split('.')
        return  tmp[0] + str(current_milli_time()) + '.' + tmp[1]
    else:
        return name + str(current_milli_time())


def showAndSaveImage(fig,file,flag):
    if flag:
        plt.show()
        fig.savefig(uniquefilename(file))
        plt.close(fig)