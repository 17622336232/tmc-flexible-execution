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


def adapt_decision_records(decision, adapter):
    def convert(container):
        records = container.get('records')
        if isinstance(records, list):
            return {**container, 'records': [adapter(record) for record in records]}
        return container

    adapted = convert(decision)
    if isinstance(decision.get('report'), dict):
        adapted = {**adapted, 'report': convert(decision['report'])}
    return adapted


def keep_observation(records, incoming):
    """Keep the latest observed field with its own image, including older complementary facts."""
    observed = facts(incoming)
    identity = object_id(incoming)
    if not observed and any(object_id(row) == identity for row in records):
        return records
    retained = []
    for row in records:
        if object_id(row) != identity:
            retained.append(row)
            continue
        remaining = {field: value for field, value in facts(row).items() if field not in observed}
        if remaining:
            retained.append({**row, 'facts': remaining})
    return retained + [incoming]


def aggregate_facts(records):
    """Combine fields from separate report photos and expose conflicting observations."""
    grouped = {}
    conflicts = []
    for row in records:
        identity = object_id(row)
        observed = grouped.setdefault(identity, {})
        for field, value in facts(row).items():
            if value is None:
                continue
            if field in observed and observed[field] != value:
                conflicts.append((identity, field))
            observed[field] = value
    return grouped, conflicts
