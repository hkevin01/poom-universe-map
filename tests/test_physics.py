import torch
from poom.physics import gravitational_acceleration

def test_self_interaction_zero():
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    m = torch.tensor([1.0, 1.0])
    a = gravitational_acceleration(pos, m, G=1.0, softening=0.1, box_size=10.0)
    assert torch.isfinite(a).all()
    # Accelerations should be opposite on two-body line
    assert torch.sign(a[0,0]) != torch.sign(a[1,0])
