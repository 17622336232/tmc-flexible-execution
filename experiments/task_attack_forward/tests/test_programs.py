"""Counterexamples for semantic shortcuts that previously inflated forecasts."""
import copy
import json
from pathlib import Path
import random
import unittest

from experiments.task_attack.tasks import make_task
from experiments.task_attack_forward.programs import answer,semantic_effect,known_program

ROOT=Path(__file__).resolve().parents[3]
PROGRAMS=json.loads((ROOT/'research/attack_learning/programs/equipment.json').read_text())['programs']


class CompositionalPrograms(unittest.TestCase):
    def setUp(self):
        self.world=[dict(cracked=False,temperature=34,display_text='EQ-01'),dict(cracked=True,temperature=86,display_text='EQ-02'),
                    dict(cracked=True,temperature=51,display_text='EQ-03'),dict(cracked=False,temperature=78,display_text='EQ-04')]
        self.stations=[dict(index=i+1,identity='EQ-%02d'%(i+1),cracked=None,temperature=None) for i in range(4)]
        self.parameters=dict(temperature_threshold=70)

    def program(self,family):return next(p for p in PROGRAMS if p['family']==family)

    def effect(self,family,action,target=None):
        return semantic_effect(PROGRAMS,[self.world],self.parameters,dict(action_id=action),self.stations,(self.program(family),target))

    def test_no_effect_on_nonminimum_is_not_an_integrity_attack(self):
        self.assertEqual(self.effect('thermal_minimum','place_decal_display_95_02'),0.)
        self.assertEqual(self.effect('thermal_minimum','place_decal_display_05_02'),1.)

    def test_conjunction_preserves_other_operand(self):
        self.assertEqual(self.effect('crack_and_heat','place_decal_shell_smooth_03'),0.)
        self.assertEqual(self.effect('crack_and_heat','place_decal_shell_smooth_02'),1.)

    def test_unselected_source_does_not_change_target_answer(self):
        self.assertEqual(self.effect('locate','place_decal_display_05_02',0),0.)
        self.assertEqual(self.effect('locate','place_decal_display_05_01',0),1.)

    def test_current_literal_labels_do_not_inherit_fact_attack_error(self):
        self.assertEqual(self.effect('equipment_labels','place_decal_shell_cracked_01'),0.)

    def test_actual_rgb_conditions_world_particles_without_world_lookup(self):
        stations=copy.deepcopy(self.stations);stations[0]['cracked']=False
        self.assertEqual(semantic_effect(PROGRAMS,[self.world],self.parameters,dict(action_id='place_decal_shell_smooth_01'),stations,(self.program('cracks'),None)),0.)
        self.assertEqual(semantic_effect(PROGRAMS,[self.world],self.parameters,dict(action_id='place_decal_shell_cracked_01'),stations,(self.program('cracks'),None)),1.)

    def test_authorized_language_diagnostic_uses_language_only(self):
        devices=[dict(identity=s['identity'],**f) for s,f in zip(self.stations,self.world)]
        for program in PROGRAMS:
            task,_,_=make_task(program['family'],devices,random.Random(94))
            found,target=known_program(task,PROGRAMS,self.stations)
            self.assertEqual(found['family'],program['family'])
            if program['scope']=='target':self.assertEqual(self.stations[target]['identity'],'EQ-02')

    def test_entire_report_is_compared_including_requested_numeric_value(self):
        self.assertEqual(self.effect('temperature','place_decal_display_95_02'),1.)
        program=self.program('temperature');report=answer(program,self.world,self.parameters)
        self.assertEqual(report,((1,(('temperature',86),)),(3,(('temperature',78),))))


if __name__=='__main__':unittest.main()
