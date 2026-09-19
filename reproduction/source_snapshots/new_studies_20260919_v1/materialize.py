"""Copy available original source layout into a new directory; never execute it."""
from pathlib import Path
import argparse
import json
import shutil
import verify

def materialize(repo_root, output):
    summary = verify.verify(repo_root)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Output must be a new directory, even if an existing one is empty')
    # No output is created until the complete source snapshot has passed verification.
    output.mkdir(parents=False, exist_ok=False)
    targets = {}
    excluded = []
    for row in verify.read('source_mapping.json')['bindings']:
        if row['content_location'] == 'external_not_in_snapshot':
            excluded.append({'logical_source_path':row['logical_source_path'],
                             'original_sha256':row['original_sha256'], 'category':row['category']})
            continue
        relative = row['logical_source_path']
        if relative in targets:
            if targets[relative] != row['original_sha256']:
                raise ValueError('Conflicting duplicate target')
            continue
        target = verify.safe_path(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        source = verify.source_path(row, repo_root)
        with source.open('rb') as reader, target.open('xb') as writer:
            shutil.copyfileobj(reader, writer)
        if verify.digest(target) != row['original_sha256']:
            raise ValueError('Copied source bytes differ')
        targets[relative] = row['original_sha256']
    report = {'schema':'source_layout_materialization_only_v1',
              'source_files_copied':len(targets), 'source_sha256_by_relative_path':targets,
              'external_dependencies_not_fabricated':excluded,
              'production_registrations_or_receipts_created':False,
              'resource_leases_created':False, 'models_or_oracles_executed':False,
              'ready_to_run_original_experiment_CLI':False}
    with (output/'MATERIALIZATION.json').open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    (output/'README.NOT_EXECUTABLE.md').write_text(
        '# Original source layout only\n\n'
        'Source bytes were copied after SHA verification. No experiment was run. '
        'External weights/data, original configuration ancestry, resource adapters and receipts are not supplied. '
        'Do not invoke original experiment CLIs or treat this directory as an accepted production workspace.\n')
    return dict(summary, materialized_source_files=len(targets), output_created=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(materialize(args.repo_root,args.output),sort_keys=True))

if __name__=='__main__':
    main()
