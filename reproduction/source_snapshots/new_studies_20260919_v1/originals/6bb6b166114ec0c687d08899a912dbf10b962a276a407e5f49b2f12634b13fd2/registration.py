"""CLI for a new immutable registration and CPU-only launch/ES evidence audit."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'frozen_src'))
import argparse,json
from matdiscovery.support_scope import (build_registration,read_registration,validate_registration,
    validate_launch,runtime_inventory,external_inventory)
from matdiscovery.immutable_hash_cache import immutable_hash_cache


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','validate','plan','audit-es'))
    p.add_argument('--workspace',required=True);p.add_argument('--parent');p.add_argument('--validation');p.add_argument('--component-registration')
    a=p.parse_args()
    with immutable_hash_cache():
        if a.mode=='prepare':
            if not a.parent or not a.validation or not a.component_registration:p.error('prepare needs --parent, --validation and --component-registration')
            reg=build_registration(a.parent,a.workspace,validation=a.validation,component_registration=a.component_registration)
            result=validate_launch(reg)

        else:
            reg=read_registration(a.workspace)
            if a.mode=='plan':result={'fingerprint':reg['fingerprint'],'ES':reg['ES_expected_counts'],'evaluation':reg['expected_counts'],'evaluation_seeds':reg['evaluation_seeds'],'science_started':False}
            elif a.mode=='validate':result=validate_launch(reg)
            else:
                validate_launch(reg)
                from full_es_training import verify_full_es_receipt
                result=verify_full_es_receipt(registration=reg)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
