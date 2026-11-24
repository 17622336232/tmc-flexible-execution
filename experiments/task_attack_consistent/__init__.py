"""Frozen-parent value policy with comparable immediate/future estimation."""
import importlib
import sys

# These remain exact frozen parent implementations; this variant changes only
# the prospective value comparison, not task execution, sensors or scoring.
for _name in ('cleanup','perception','programs','scene','vision'):
    sys.modules[__name__+'.'+_name]=importlib.import_module('experiments.task_attack_value.'+_name)
