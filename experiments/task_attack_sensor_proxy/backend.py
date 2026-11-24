"""Report a genuinely latched native stop instead of retrying impossible goals.

Original planner/controller protections are untouched. This changes experiment
error reporting only; normal replanning warnings remain ordinary tool feedback.
"""
from experiments.task_attack.runtime import GazeboBackend as OriginalBackend


class NativeSensorStop(RuntimeError):
    code='native_sensor_timeout_latched'


class GazeboBackend(OriginalBackend):
    def __init__(self,*args,**kwargs):
        self.native_sensor_stop=None
        super().__init__(*args,**kwargs)

    def _planner_log(self,message):
        if message.name=='/vla_planner' and 'Depth Lost! EMERGENCY_STOP' in message.msg:
            self.native_sensor_stop=message.msg
            self.event('native_sensor_timeout_latched',message=message.msg)
        super()._planner_log(message)

    def observe(self):
        if self.native_sensor_stop:
            raise NativeSensorStop('Original planner latched its depth-loss safety stop; no reset or further autonomous goal is attempted: '+self.native_sensor_stop)
        return super().observe()

    def _native_failure_since(self,started):
        return self.native_sensor_stop or super()._native_failure_since(started)
