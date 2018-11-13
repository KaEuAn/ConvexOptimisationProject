from oracles import FirstOrderOracle as first_order_oracle
from oracles import SecondOrderOracle as second_order_oracle
from steps import SimpleStep as step_size

class Task1:
    def __init__(a, b, F, g):
        self.first_order_oracle = first_order_oracle(a, b)
        self.second_order_oracle = second_order_oracle(a, b)
        self.step_size = step_size()
        self.
        