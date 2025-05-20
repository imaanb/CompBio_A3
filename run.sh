#!/bin/bash

print("This will take approximately 2-3 hours")
print("Starting first simulation out of 3")
python SIR_ABM_macrophages.py
print("Finished first simulation, starting second")
python SIR_simulate.py --macrophages 0
print("Finished second simulation starting third")
python SIR_simulate.py --macrophages 50
print("Finished simulations, plotting")
python macrophages_performance_plot.py
python SIR_plot.py
