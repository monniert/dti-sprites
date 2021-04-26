from torch.optim import SGD, Adam, AdamW, ASGD, Adamax, Adadelta, Adagrad, RMSprop


def get_optimizer(name):
    if name is None:
        name = 'sgd'
    return {
        "sgd": SGD,
        "adam": Adam,
        "adamw": AdamW,
        "asgd": ASGD,
        "adamax": Adamax,
        "adadelta": Adadelta,
        "adagrad": Adagrad,
        "rmsprop": RMSprop,
    }[name]


class MultipleOptimizer:
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
