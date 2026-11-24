"""Finite operation/trace belief pilot; no privileged current-task inputs."""
import importlib
import sys

for _name in ('cleanup','perception','programs','scene','vision'):
    sys.modules[__name__+'.'+_name]=importlib.import_module('experiments.task_attack_value.'+_name)
