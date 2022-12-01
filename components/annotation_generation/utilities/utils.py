import multiprocessing as mp


def num_workers(multiprocessing: object = False) -> object:
    """

    Args:
        multiprocessing:

    Returns:

    """
    if not multiprocessing:
        return 1
    elif mp.cpu_count() == 1:
        return 1
    else:
        return mp.cpu_count()-1