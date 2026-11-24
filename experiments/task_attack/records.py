"""Task-neutral report structure and legacy report compatibility."""


def object_id(row):
    return row.get('object_id', row.get('equipment_id')) if isinstance(row, dict) else None


def facts(row):
    result = dict(row.get('facts', {})) if isinstance(row.get('facts'), dict) else {}
    for key in ('cracked', 'temperature'):
        if key in row and key not in result:result[key] = row[key]
    return result


def validate_record(row, evidence_ids):
    if not isinstance(row, dict) or not isinstance(object_id(row), str) or not object_id(row).strip():
        raise ValueError('A record needs a nonempty object_id')
    if not isinstance(row.get('facts', {}), dict):raise ValueError('Record facts must be an object')
    if row.get('evidence_image_id') not in evidence_ids:
        raise ValueError('Record refers to an image that was never captured')
