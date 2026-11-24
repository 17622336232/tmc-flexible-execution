"""External-observation task-aware attack pilot (zero-shot implementation)."""
import os

POLICIES = ('clean', 'blind', 'inferred', 'known_task')
FAMILIES = ('cracks', 'temperature', 'locate', 'inventory', 'color_search',
            'order_check', 'parking', 'blocked_exit', 'fire_safety',
            'crack_and_heat', 'thermal_minimum', 'safe_equipment',
            'shortage', 'overstock', 'large_inventory',
            'vehicle_search', 'clear_exit', 'fire_and_access',
            'equipment_labels', 'warehouse_labels', 'campus_labels',
            'locate_crack','locate_condition','locate_boxes','locate_order',
            'locate_extinguisher','locate_access')
RESEARCH_POLICIES = POLICIES + ('belief', 'belief_blind', 'belief_known_task', 'uao_spsa', 'iccv_uada_spsa',
                              'iccv_upa_spsa', 'redvla_external')
MODEL = os.environ.get('UAV_TASK_MODEL', 'gpt-6.1-terra')
EFFORT = 'max'
ATTACK_EFFORT = 'high'
ATTACK_EFFORTS = ('high', 'xhigh', 'max')
SOURCE_PROFILES = ('single', 'independent', 'shared')
