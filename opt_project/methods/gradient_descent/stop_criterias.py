from ...core.abstract_stop_criterias import xdiff_stop_crit as x_s_c
from ...core.abstract_stop_criterias import ydiff_stop_crit as y_s_c
from ...core.abstract_stop_criterias import iteration_stop_crit as it_s_c

class xdiff_stop_crit(x_s_c):
    def __call__(self, x, *args, **kwargs):
        return super().__call__(x)

class ydiff_stop_crit(y_s_c):
    def __call__(self, x, oracle, *args, **kwargs):
        return super().__call__(oracle.func(x))

class iteration_stop_crit(it_s_c):
    def __call__(self, *args, **kwargs):
        return super().__call__()