import multiprocessing as mp


def num_workers(multiprocessing=False):

    if not multiprocessing:
        return 1
    elif mp.cpu_count() == 1:
        return 1
    else:
        return mp.cpu_count()-1