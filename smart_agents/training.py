"""Immutable dataset helpers shared by training and readiness checks."""
import hashlib
import json
from pathlib import Path


def write_new_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def split_groups(rows, *, assignments=None):
    """Keep source/session/lesson variants together, with stable component hashing.

    Include lesson_group_ids for reviewed paraphrase groups. Shared candidate
    lesson IDs also join components; insufficient independent components must
    be addressed in dataset construction, never by leaking evidence.
    """
    assignments = assignments if assignments is not None else {}
    keys_by_row = []
    parents = list(range(len(rows)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        keys = [('lesson', str(row['task_id']))] if 'task_id' in row else [('task', str(row.get('task','')))]
        for field in ('group_id', 'source_session_id'):
            if row.get(field): keys.append((field, str(row[field])))
        keys += [('lesson', str(c['id'])) for c in row.get('candidates', [])]
        for c in row.get('candidates', []):
            if c.get('source_session_id'):keys.append(('source_session_id',str(c['source_session_id'])))
        keys += [('paraphrase', str(x)) for x in row.get('lesson_group_ids', [])]
        keys_by_row.append(keys)
        for key in keys:
            if key in seen: parents[root(i)] = root(seen[key])
            else: seen[key] = i
    components = {}
    for i, row in enumerate(rows):components.setdefault(root(i), []).append((row,keys_by_row[i]))
    result = {'train': [], 'validation': [], 'release': []}
    for component in components.values():
        keys = sorted({json.dumps(key) for _,ks in component for key in ks})
        previous = {assignments[key] for key in keys if key in assignments}
        if len(previous)>1:raise ValueError('New evidence joins previously separate partitions; rebuild a reviewed dataset without leakage')
        if previous:partition=next(iter(previous))
        else:
            bucket=int(hashlib.sha256(keys[0].encode()).hexdigest()[:8],16)%10
            partition='release' if bucket==9 else 'validation' if bucket==8 else 'train'
        if partition not in result:raise ValueError('Invalid persisted partition')
        for key in keys:assignments[key]=partition
        result[partition].extend(row for row,_ in component)
    return result
