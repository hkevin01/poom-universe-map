import torch
from poom.initial_conditions import make_ic
from poom.integrators import leapfrog_step

def test_leapfrog_runs():
    pos, vel, m = make_ic(N=16, box_size=10.0, seed=0)
    pos2, vel2 = leapfrog_step(pos.clone(), vel.clone(), m, dt=0.01, G=1.0, softening=0.1, box_size=10.0)
    assert pos2.shape == pos.shape
    assert vel2.shape == vel.shape
