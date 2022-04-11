import numpy as np
import pandas as pd
from spinneret import *

xes = np.arange(0,100)
sin1 = np.sin(xes / 4)
sin2 = np.sin(xes / 4.05)
sin3 = np.sin(xes / 4.5)
sin4 = np.sin(xes / 4) / 3

y_vals = np.random.randn(10000)
print(rms(y_vals, np.zeros(10000)))


# sin_model = model(xes, sin1)

def test_rms():

    assert rms(sin1, sin1) == 0
    assert rms(sin1, sin2) < rms(sin1, sin3)

    # ass


def test_mad():

    assert mad(sin1, sin1) == 0
    assert mad(sin1, sin2) < mad(sin1, sin3)

def test_rvar():

    assert rvar(sin1) > rvar(sin4)
