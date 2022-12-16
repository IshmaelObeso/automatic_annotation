import multiprocessing as mp


def num_workers(multiprocessing: bool = False) -> int:
    """
    Calculates how many processes to assign based on whether we should multiprocess or not

    Args:
        multiprocessing: bool that determines if we should multiprocess or not

    Returns:
        int: if multiprocessing is False, always return 1, if multiprocessing is True, return number_of_cores-1

    """

    if not multiprocessing:
        return 1
    elif mp.cpu_count() == 1:
        return 1
    else:
        return mp.cpu_count()-1