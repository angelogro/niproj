#!/usr/bin/env python
# coding: utf-8

import sys

from agent.traincatan import TrainCatan

train=None

def pairwise(it):
    it = iter(it)
    while True:
        yield next(it), next(it)

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def isint(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def istuple(value):
    try:
        tuple(value)
        eval(value)
        return True
    except (ValueError,SyntaxError,NameError):
        return False

if __name__ == "__main__":

    train = TrainCatan(print_episodes=True)

    for param,param_value in pairwise(sys.argv[2:]): #first argument is executed file,second argument is instance name
        if isint(param_value):
            setattr(train,param,int(param_value))
        elif isfloat(param_value):
            setattr(train,param,float(param_value))
        elif istuple(param_value):
            setattr(train,param,eval(param_value))
        else:
            setattr(train,param,param_value)

    train.start_training()

    train.save_hyperparameters(sys.argv[1])

