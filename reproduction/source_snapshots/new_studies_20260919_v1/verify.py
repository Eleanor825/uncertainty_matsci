"""Verify stored source bytes and referenced repository files; never import them."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json

HERE = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def safe_path(root, relative):
    p = PurePosixPath(relative)
    if not relative or p.is_absolute() or '..' in p.parts or '\\' in relative:
        raise ValueError('Expected a safe repository-relative path')
    candidate = Path(root).resolve() / p
    if not candidate.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes source root')
    return candidate

def read(name):
    return json.loads((HERE / name).read_text())

def source_path(row, repo_root):
    if row['content_location'] == 'snapshot':
        return safe_path(HERE, row['content_path'])
    if row['content_location'] == 'existing_repository':
        return safe_path(repo_root, row['content_path'])
    raise ValueError('Binding has no included source payload')

def verify(repo_root):
    repo_root = Path(repo_root).resolve()
    manifest = read('manifest.json')
    expected = {x['path'] for x in manifest['files']}
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    if actual != expected:
        raise ValueError('Snapshot file set differs from manifest')
    for item in manifest['files']:
        path = safe_path(HERE, item['path'])
        if path.stat().st_size != item['bytes'] or digest(path) != item['export_sha256']:
            raise ValueError('Snapshot bytes differ: ' + item['path'])
    payloads = read('payload_index.json')['payloads']
    if len(payloads) != 61 or len({p['original_sha256'] for p in payloads}) != 61:
        raise ValueError('Expected the approved61 unique original payloads')
    for item in payloads:
        if digest(safe_path(HERE, item['snapshot_path'])) != item['original_sha256']:
            raise ValueError('Original source was changed')
    mapping = read('source_mapping.json')
    included = []
    excluded = []
    for row in mapping['bindings']:
        if row['content_location'] == 'external_not_in_snapshot':
            excluded.append(row)
            continue
        path = source_path(row, repo_root)
        if not path.is_file() or digest(path) != row['original_sha256']:
            raise ValueError('Registered source mismatch: ' + row['logical_source_path'])
        included.append(row)
    targets = {}
    for row in included:
        relative = row['logical_source_path']
        safe_path(HERE, relative)  # lexical traversal guard; no directory creation
        if relative in targets and targets[relative] != row['original_sha256']:
            raise ValueError('Conflicting original bytes at one logical path')
        targets[relative] = row['original_sha256']
    projections = read('configuration_projections/index.json')['projections']
    for p in projections:
        path = safe_path(HERE, p['path'])
        if digest(path) != p['export_sha256']:
            raise ValueError('Derived configuration differs')
        value = json.loads(path.read_text())
        if value['original_reference']['sha256'] != p['original_sha256'] or value['not_an_original_registration'] is not True:
            raise ValueError('Derived configuration misrepresented as original')
    return {'passed': True, 'original_source_payloads': len(payloads),
            'included_source_bindings': len(included), 'distinct_logical_source_files': len(targets),
            'explicit_external_bindings': len(excluded), 'derived_configuration_projections':len(projections),
            'model_or_oracle_calls':0, 'original_CLI_ready_on_another_machine':False}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.repo_root), sort_keys=True))

if __name__ == '__main__':
    main()
