import time
from multiprocessing.pool import MapResult


class Progressbar:

    def __init__(self, total: int, bar_length: int = 50, title: str = None) -> None:
        if title is None:
            title = ""
        else:
            title += " "
        self.__total = total
        self.__bar_length = bar_length
        self.__title = title

    def update(self, iteration: int):
        if iteration > self.__total:
            raise ValueError("Iteration is out of range")
        progress = iteration / self.__total
        arrow = '=' * int(round(self.__bar_length * progress))
        spaces = ' ' * (self.__bar_length - len(arrow))
        percent = int(progress * 100)
        print(f'{self.__title}[{arrow + spaces}] {percent}% complete', end='\r')
        if iteration >= self.__total:
            print()

    def update_multiprocessing(self, pool_result: MapResult, sleep: float = 0.5) -> None:
        finished_count = 0
        pool_result_values = pool_result._value
        while finished_count < self.__total:
            finished_count = sum(1 for result in pool_result_values if result)
            self.update(finished_count)
            time.sleep(sleep)
