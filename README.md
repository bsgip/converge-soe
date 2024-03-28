# converge-soe: SOE optimisation engine as used in Project Converge

converge-soe is a python package that implements the shared operating envelope (SOE) optimisation as used in [Project Converge](https://arena.gov.au/projects/project-converge-act-distributed-energy-resources-demonstration-pilot/). It is an open-source version of the code that was tested during the project.

## Prerequisites
`converge-soe` requires the presence of the [IPOPT library](https://coin-or.github.io/Ipopt/). Installation instructions for IPOPT are provided [here](https://coin-or.github.io/Ipopt/INSTALL.html). Note that `converge-soe` requires that the optional ASL library is installed, as explained in the IPOPT installation instructions. A linear solver library must also be installed - both MUMPS and MA27 have been tested with `converge-soe`. The instructions provide links for both of these. MUMPS is fully open source, while MA27 is available free for research but requires a special licence.

## Installation
1. Install IPOPT (see above)
1. Make sure the `ipopt` executable is on your path.
1. Install the converge-soe package 
```
pip3 install -e .
```

## Use
```python
import converge_soe as csoe
from pyomo.opt import SolverStatus

solver = csoe.SoeSolver(
    netw, forecast, offers, envelope_abs_max=50.0, solver_options={'linear_solver': 'ma27'}
)
status, results = solver.solve()
is_ok = status == SolverStatus.ok
print(results.bus)
print(results.branch)
print(results.soe)
print(results.viol)
```

See the code in `examples/run_scenario.py` for a full example. This is invoked as follows:
```python
cd examples
python ./run_scenario.py -s 1.0 scenario_1 scenario_1_output
```
This will run the SOE engine, using input files in the `scenario_1` directory. Following successful completion, various output CSV and log files will be found in a newly created `scenario_1_output` directory.

The `-s 1.0` part of the command above scales all loads by a factor of 1.0 i.e. it keeps them unmodified. In `example_1`, the network is not overloaded and hence envelopes are permissive. We can see this by looking at `scenario_1_output/soe.csv`, which looks like this:
```csv
load_id,soe_lb_kw,soe_ub_kw,dispatch_kw,payment_dlr
load_1118,-49.999975,49.999975,0.0,0.0
load_1131,-49.999975,49.999975,0.0,0.0
...
```
Most loads have envelopes between -50 kW and 50 kW, and there is no network support dispatch.

We can induce a network support response and more restrictive envelopes by scaling the loads in the system - for example, by a factor of 2:
```python
python ./run_scenario.py -s 2.2 scenario_1 scenario_1_output
```
This produces the following output:
```
load_id,soe_lb_kw,soe_ub_kw,dispatch_kw,payment_dlr
load_1118,-49.999907,49.999991,0.0,0.0
...
load_541,-2.322306,49.999991,0.0,0.0
load_614,-4.445983,49.999991,0.0,0.0
...
```
We see that there are more restrictive lower (load) envelopes, but still no network support dispatches. Scaling the loads further, to 2.5, produces even more restrictive lower envelopes (zero), and includes generation network support dispatch to offset the excessive load:
```
load_id,soe_lb_kw,soe_ub_kw,dispatch_kw,payment_dlr
load_1118,0.0,49.999975,0.0,0.0
...
load_614,0.0,49.999975,3.247845,0.270654
load_632,0.0,49.999975,0.498675,0.041556
load_644,-2.625459,49.999975,0.0,0.0
...
```
