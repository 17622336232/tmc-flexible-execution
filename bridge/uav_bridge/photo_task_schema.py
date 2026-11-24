"""Open photographic requirements and ordered captures, shared by service/runner.

The model supplies the shot grouping. This module checks references and bounds;
it contains no viewpoint, pose, object or language-specific task catalogue.
"""
import copy


def requirements(task):
    items = task.get('requirements')
    if (task.get('schema_version') not in (2, 3) or
            any(k in task for k in ('view', 'framing', 'trigger', 'subject')) or
            not isinstance(items, list) or not 1 <= len(items) <= 12):
        raise ValueError('摄影任务需要完整的开放要求格式。')
    for index, item in enumerate(items, 1):
        if (not isinstance(item, dict) or set(item) != {'id', 'text', 'source_text'} or
                item['id'] != 'r%d' % index or
                not isinstance(item['text'], str) or not item['text'].strip() or len(item['text']) > 1000 or
                not isinstance(item['source_text'], str) or not item['source_text'].strip() or
                len(item['source_text']) > 2000 or item['source_text'] not in task.get('user_request', '')):
            raise ValueError('摄影要求编号、内容或原话引用无效，不能删减已确认要求。')
    return items


def shots(task):
    items = requirements(task)
    if task['schema_version'] == 2:
        return [dict(id='s1', summary=task.get('summary', '拍摄照片'),
                     requirement_ids=[item['id'] for item in items])]
    sequence = task.get('shots')
    if not isinstance(sequence, list) or not 1 <= len(sequence) <= 6:
        raise ValueError('一次摄影任务需要包含 1 到 6 张按顺序拍摄的照片。')
    expected, used = {item['id'] for item in items}, set()
    for index, shot in enumerate(sequence, 1):
        if (not isinstance(shot, dict) or set(shot) != {'id', 'summary', 'requirement_ids'} or
                shot['id'] != 's%d' % index or not isinstance(shot['summary'], str) or
                not shot['summary'].strip() or len(shot['summary']) > 500 or
                not isinstance(shot['requirement_ids'], list) or not 1 <= len(shot['requirement_ids']) <= 12 or
                any(not isinstance(key, str) or key not in expected for key in shot['requirement_ids']) or
                len(set(shot['requirement_ids'])) != len(shot['requirement_ids'])):
            raise ValueError('照片顺序、描述或要求引用无效。')
        used.update(shot['requirement_ids'])
    if used != expected:
        raise ValueError('照片序列遗漏了已确认的摄影要求。')
    return sequence


def shot_requirements(task, index=0):
    sequence = shots(task)
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(sequence):
        raise ValueError('当前照片编号无效，不能跳过或重复拍摄。')
    ids = sequence[index]['requirement_ids']
    return [copy.deepcopy(item) for item in requirements(task) if item['id'] in ids]
