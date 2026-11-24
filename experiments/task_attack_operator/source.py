"""Reuse offline geometric labels for co-located new physical carriers.

No current target camera, source contents or simulator pose is available here.
The older shutter and new decals share an actual surface footprint. Their
appearance/acceptance effects are NOT assumed equal by this geometric model.
"""
import re
from experiments.task_attack_cached.source_emission import source_probability as original


def source_probability(prediction,action_id):
    match=re.fullmatch(r'place_decal_(shell|display)_(?:smooth|cracked|35|95)_(\d+)',action_id)
    source=('cover_%s_%s'%(match[1],match[2])) if match else action_id
    return original(prediction,source)
