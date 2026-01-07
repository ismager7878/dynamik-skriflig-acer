import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import pprint

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'scripts'))

import importlib
import opg2
importlib.reload(opg2)
from opg2 import s_vec

def find_forces_lagrange(type: str, vels, link_lengths):

    #Get length of type string
    axisCount = len(type)

    forceList = []

    for i in range(axisCount):
        pass
    return forceList


def inertia(mass, link_length, type = "rod", com_dist = 0):
    if("rod"):
        return (mass*link_length**2)/12

def kin_energy(type, ang_vel, trans_vel, mass, inertia):
    if(type == 'g'):
        return 1/2 * mass * trans_vel.dot(trans_vel) + 1/2 * () * ang_vel**2
    
        

    return T