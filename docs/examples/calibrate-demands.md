# Calibrating demands on an EPANET network

**Goal.** Three junctions of EPANET's Example Network 1 have demands nobody has metered.
Pressure loggers at six junctions record the heads. Find the three demands that reproduce
the logged heads. The solve is differentiable, so the gradient of the misfit with respect to
every demand comes from one backward pass and an off-the-shelf optimiser (L-BFGS) does the rest.
There is no finite-difference loop and no re-linearisation by hand.

The "measurements" here are synthetic: heads computed from known demands. That makes the
answer checkable, and it is how you should test any calibration before you trust it on real
data. [Calibrate with gradients](../get-started/calibrate.md) introduces the same idea on the
smallest possible model.

```python
import torch
from noodl.apps.water import read_epanet_inp, build_model, water_steady, initial_drivers

net = read_epanet_inp("Net1.inp")
model, state, _ = build_model(net)
water = model.refs.water

def heads(demand):
    """Steady head (m) at every node for `demand`, a {junction: m3/s withdrawn} mapping."""
    drivers = initial_drivers(model, demand=demand)
    return water.head.named(water_steady(model, state, drivers)[water.head])

unknown = ["12", "21", "31"]                       # junctions with unmetered demand
sensors = ["11", "12", "13", "22", "23", "32"]     # junctions with a pressure logger

truth = {"12": 0.012, "21": 0.020, "31": 0.016}    # m3/s, used only to make the data
measured = heads(truth)
target = torch.stack([measured[s] for s in sensors])

guess = torch.full((3,), 0.008, dtype=torch.float64, requires_grad=True)

def misfit(demands):
    h = heads(dict(zip(unknown, demands)))
    return (((torch.stack([h[s] for s in sensors]) - target) * 1000.0) ** 2).sum()  # mm^2

# One backward pass gives the sensitivity of the misfit to every demand.
first = misfit(guess)
first.backward()
print("misfit at the first guess (mm^2):", round(first.item(), 1))
print("gradient (mm^2 per m3/s):", [f"{g:.3e}" for g in guess.grad.tolist()])

optimiser = torch.optim.LBFGS(
    [guess], max_iter=50, tolerance_grad=1e-14, tolerance_change=1e-16,
    line_search_fn="strong_wolfe",
)

def closure():
    optimiser.zero_grad()
    loss = misfit(guess)
    loss.backward()
    return loss

for _ in range(2):
    optimiser.step(closure)
print("fitted demands (L/s):", [round(1000 * float(d), 3) for d in guess])
print("true demands (L/s):  ", [round(1000 * truth[j], 3) for j in unknown])
```

```text
misfit at the first guess (mm^2): 10441791.2
gradient (mm^2 per m3/s): ['-8.709e+06', '-4.915e+08', '-1.283e+09']
fitted demands (L/s): [12.0, 20.0, 16.0]
true demands (L/s):   [12.0, 20.0, 16.0]
```

![Estimated demands per iteration approaching the true values, and the head misfit falling](../assets/examples/calibrate-demands.svg#only-light)
![Estimated demands per iteration approaching the true values, and the head misfit falling](../assets/examples/calibrate-demands-dark.svg#only-dark)

The first trial steps of the line search overshoot far outside the plotted range, which is cut
at 40 L/s. The optimiser recovers all three demands to well below a millilitre per second. Heads in Net1
are held by a reservoir and a tank, so they react weakly to demand: a change of a few litres per
second moves a head by about a millimetre. That is why the misfit is scaled to millimetres; without it the
default tolerances of an optimiser would stop it at a visibly wrong answer.

**Things to try.** Drop sensors until the problem becomes under-determined, and see which
demands the remaining data can no longer tell apart. Add noise to `target` and see how far
the fitted demands move. The same recipe fits a pump speed or a valve setting (given as
drivers) and the reaction coefficients of the water-quality model; see
[Gradients](../guide/gradients.md) for which inputs a gradient can reach and
[Water distribution](../applications/water/index.md) for the application.
