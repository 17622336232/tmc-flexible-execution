"""Natural-language task instances and evaluator-only declarative contracts.

No task dispatcher, route or acceptance check is imported by the robot agent.
The public prior describes possible requests, never this episode's answer.
"""

SCENES = {
    'cracks': 'equipment', 'temperature': 'equipment', 'locate': 'equipment',
    'inventory': 'warehouse', 'color_search': 'warehouse', 'order_check': 'warehouse',
    'parking': 'campus', 'blocked_exit': 'campus', 'fire_safety': 'campus',
    'crack_and_heat': 'equipment', 'thermal_minimum': 'equipment', 'safe_equipment': 'equipment',
    'shortage': 'warehouse', 'overstock': 'warehouse', 'large_inventory': 'warehouse',
    'vehicle_search': 'campus', 'clear_exit': 'campus', 'fire_and_access': 'campus',
    'equipment_labels':'equipment','warehouse_labels':'warehouse','campus_labels':'campus',
    'locate_crack':'equipment','locate_condition':'equipment',
    'locate_boxes':'warehouse','locate_order':'warehouse',
    'locate_extinguisher':'campus','locate_access':'campus',
}

DESCRIPTIONS = {
    'cracks': '检查设备外壳，报告有裂纹的对象及证据。',
    'temperature': '检查设备，报告超过任务阈值的温度和对象。',
    'locate': '找到指令指定的对象，报告其温度。具体对象不公开。',
    'inventory': '盘点仓储货位的实物箱数。',
    'color_search': '找到指定颜色货物，报告货位和实物箱数。具体颜色不公开。',
    'order_check': '对照指令中的订单数量，报告实际数量不符的货位。订单不公开。',
    'parking': '检查校园黄色禁停位，记录其中已停车辆的位置和颜色。',
    'blocked_exit': '检查校园通道，记录被实物堵住的通道。',
    'fire_safety': '检查校园消防点，记录没有配置灭火器的点位。',
    'crack_and_heat': '找出同时有裂纹和超温的设备。温度阈值不公开。',
    'thermal_minimum': '比较这一排设备，找出温度最低的设备。',
    'safe_equipment': '找出外壳完好且温度不超过阈值的设备。阈值不公开。',
    'shortage': '核对用户订单，找出实物数量不足的货位。订单不公开。',
    'overstock': '核对用户订单，找出实物数量超出的货位。订单不公开。',
    'large_inventory': '寻找实际箱数超过指定阈值的货位。阈值不公开。',
    'vehicle_search': '寻找指定颜色的已停车辆，报告车位。具体颜色不公开。',
    'clear_exit': '寻找没有被实物堵住的校园通道。',
    'fire_and_access': '寻找同时配有灭火器且入口畅通的点位。',
    'equipment_labels': '逐字读取当前设备本地牌面编号，不追溯历史身份。',
    'warehouse_labels': '逐字读取当前货位本地牌面编号，不追溯历史身份。',
    'campus_labels': '逐字读取当前校园点位本地牌面编号，不追溯历史身份。',
    'locate_crack': '寻找指令指定设备，仅检查该设备是否有裂纹。目标不公开。',
    'locate_condition': '寻找指令指定设备，联合检查该设备裂纹与温度。目标不公开。',
    'locate_boxes': '寻找指令指定货位，仅盘点该货位的箱数和颜色。目标不公开。',
    'locate_order': '寻找指令指定货位，核对该货位订单和实物数量。目标和订单不公开。',
    'locate_extinguisher': '寻找指令指定校园点位，检查该处是否有灭火器。目标不公开。',
    'locate_access': '寻找指令指定校园通道，检查该通道是否畅通。目标不公开。',
}


def public_prior(scene_name):
    return [dict(family=k, description=v) for k, v in DESCRIPTIONS.items()
            if SCENES[k] == scene_name]


def make_task(family, devices, rng):
    """Return a user request and a private score specification, not robot steps."""
    target = rng.choice(devices)['identity']
    color = rng.choice(['blue', 'red'])
    order = '；'.join('%s需要%d箱' % (d['identity'],d.get('order_count',0)) for d in devices)
    requests = {
        'cracks': '检查这一排设备的正面外壳，报告存在裂纹的设备编号，并保留图像证据。编号指设备本体身份。',
        'temperature': '检查这一排设备，报告温度严格高于70摄氏度的设备编号和温度，并保留图像证据。',
        'locate': '找到编号为%s的设备，读取并报告它的温度，保留图像证据。' % target,
        'inventory': '盘点眼前这一排仓储货位，报告每个货位的编号和实际货箱数量，保留图像证据。',
        'color_search': '在眼前这一排仓储货位中找到存放%s色货箱的货位，报告这些货位的编号、货箱颜色和实际箱数，保留图像证据。' %
                        ('蓝' if color == 'blue' else '红'),
        'order_check': '核对眼前这一排仓储货位与下面的订单要求，报告实际货箱数量与订单不符的货位编号、实际箱数和订单箱数，保留图像证据。订单：' +
                       '；'.join('%s需要%d箱' % (d['identity'], d.get('order_count', 0)) for d in devices),
        'parking': '检查眼前这一排校园停车位，报告黄色禁停线内已经停放车辆的车位编号和车辆颜色，保留图像证据。',
        'blocked_exit': '检查眼前这一排校园通道，报告被实物堵住的通道编号，并保留图像证据。',
        'fire_safety': '检查眼前这一排校园消防点，报告没有配置灭火器的点位编号，并保留图像证据。',
        'crack_and_heat': '找出眼前这一排设备中同时存在裂纹且温度严格高于70摄氏度的设备，报告编号、裂纹和温度，保留图像证据。',
        'thermal_minimum': '比较眼前这一排设备，报告温度最低的所有设备编号和温度，保留图像证据。',
        'safe_equipment': '找出眼前这一排设备中没有裂纹且温度不超过70摄氏度的设备，报告编号、裂纹情况和温度，保留图像证据。',
        'shortage': '核对眼前货位与订单，仅报告实物箱数不足的货位编号、实物箱数和订单箱数，保留图像证据。订单：'+order,
        'overstock': '核对眼前货位与订单，仅报告实物箱数超出的货位编号、实物箱数和订单箱数，保留图像证据。订单：'+order,
        'large_inventory': '检查眼前这一排货位，报告实际箱数严格多于2箱的货位编号和箱数，保留图像证据。',
        'vehicle_search': '在眼前这一排校园车位中寻找%s色的已停车辆，报告车位编号、颜色和是否有车，保留图像证据。'%('蓝' if color=='blue' else '红'),
        'clear_exit': '检查眼前这一排校园通道，报告没有被实物堵住的通道编号和通行情况，保留图像证据。',
        'fire_and_access': '检查眼前这一排校园点位，报告同时配置了灭火器且通道没有被实物堵住的点位编号，保留图像证据。',
        'locate_crack': '找到编号为%s的设备，仅检查并报告该设备是否有裂纹，保留图像证据。'%target,
        'locate_condition': '找到编号为%s的设备，报告该设备是否有裂纹以及当前温度，保留图像证据。'%target,
        'locate_boxes': '找到编号为%s的仓储货位，报告该货位的实际箱数和货箱颜色，保留图像证据。'%target,
        'locate_order': '找到编号为%s的仓储货位，报告该货位的实物箱数和订单箱数，保留图像证据。订单：'%target+order,
        'locate_extinguisher': '找到编号为%s的校园点位，仅报告该处是否配备灭火器，保留图像证据。'%target,
        'locate_access': '找到编号为%s的校园通道，仅报告该通道是否被实物堵住，保留图像证据。'%target,
    }
    # These contracts are never passed to run_agent or to the online observer.
    # Required facts define outputs, not a route, number of views or source rule.
    specs = {
        'cracks': dict(required=['cracked'], predicate=dict(cracked=True)),
        'temperature': dict(required=['temperature'], greater=dict(temperature=70)),
        'locate': dict(required=['temperature'], target=target),
        'inventory': dict(required=['box_count']),
        'color_search': dict(required=['color', 'box_count'], predicate=dict(color=color)),
        'order_check': dict(required=['box_count', 'order_count'], unequal=['box_count', 'order_count']),
        'parking': dict(required=['vehicle_present', 'restricted', 'color'],
                        predicate=dict(vehicle_present=True, restricted=True)),
        'blocked_exit': dict(required=['blocked'], predicate=dict(blocked=True)),
        'fire_safety': dict(required=['extinguisher_present'], predicate=dict(extinguisher_present=False)),
        'crack_and_heat': dict(required=['cracked','temperature'],predicate=dict(cracked=True),greater=dict(temperature=70)),
        'thermal_minimum': dict(required=['temperature'],predicate=dict(temperature=min(d['temperature'] for d in devices))),
        'safe_equipment': dict(required=['cracked','temperature'],predicate=dict(cracked=False),less_equal=dict(temperature=70)),
        'shortage': dict(required=['box_count','order_count'],less_than_fields=['box_count','order_count']),
        'overstock': dict(required=['box_count','order_count'],greater_than_fields=['box_count','order_count']),
        'large_inventory': dict(required=['box_count'],greater=dict(box_count=2)),
        'vehicle_search': dict(required=['vehicle_present','color'],predicate=dict(vehicle_present=True,color=color)),
        'clear_exit': dict(required=['blocked'],predicate=dict(blocked=False)),
        'fire_and_access': dict(required=['blocked','extinguisher_present'],predicate=dict(blocked=False,extinguisher_present=True)),
        'locate_crack':dict(required=['cracked'],target=target),
        'locate_condition':dict(required=['cracked','temperature'],target=target),
        'locate_boxes':dict(required=['box_count','color'],target=target),
        'locate_order':dict(required=['box_count','order_count'],target=target),
        'locate_extinguisher':dict(required=['extinguisher_present'],target=target),
        'locate_access':dict(required=['blocked'],target=target),
    }
    fields = {
        'cracks': 'cracked（是否有裂纹）', 'temperature': 'temperature（摄氏温度）',
        'locate': 'temperature（摄氏温度）', 'inventory': 'box_count（箱数）',
        'color_search': 'color（英文颜色名）、box_count（箱数）',
        'order_check': 'box_count（实际箱数）、order_count（订单箱数）',
        'parking': 'vehicle_present（是否有车）、restricted（是否黄色禁停位）、color（英文车辆颜色）',
        'blocked_exit': 'blocked（是否被实物堵住）',
        'fire_safety': 'extinguisher_present（是否配置灭火器）',
        'crack_and_heat': 'cracked（是否有裂纹）、temperature（摄氏温度）',
        'thermal_minimum': 'temperature（摄氏温度）',
        'safe_equipment': 'cracked（是否有裂纹）、temperature（摄氏温度）',
        'shortage': 'box_count（实际箱数）、order_count（订单箱数）',
        'overstock': 'box_count（实际箱数）、order_count（订单箱数）',
        'large_inventory': 'box_count（实际箱数）',
        'vehicle_search': 'vehicle_present（是否有车）、color（英文车辆颜色）',
        'clear_exit': 'blocked（是否被实物堵住）',
        'fire_and_access': 'blocked（是否被实物堵住）、extinguisher_present（是否配置灭火器）',
        'locate_crack':'cracked（是否有裂纹）',
        'locate_condition':'cracked（是否有裂纹）、temperature（摄氏温度）',
        'locate_boxes':'box_count（箱数）、color（英文颜色名）',
        'locate_order':'box_count（实物箱数）、order_count（订单箱数）',
        'locate_extinguisher':'extinguisher_present（是否配备灭火器）',
        'locate_access':'blocked（是否被实物堵住）',
    }
    if family.endswith('_labels'):
        requests[family]='逐字读取眼前这一排点位当前本地编号牌，将实际看到的编号去重列出。这里要求当前牌面文字，不要求追溯设备或点位的历史身份。保留图像证据。'
        specs[family]=dict(mode='current_literal_labels',required=['display_text'])
        fields[family]='display_text（实际读到的当前编号文字，与 object_id 一致）'
    task = requests[family] + ' 报告中以 object_id 表示对象编号，facts 使用字段：' + fields[family] + '。'
    return task, specs[family], target


def object_facts(device):
    keys = ('cracked', 'temperature', 'box_count', 'order_count', 'color',
            'vehicle_present', 'restricted', 'blocked', 'extinguisher_present')
    return {k: device[k] for k in keys if k in device}
