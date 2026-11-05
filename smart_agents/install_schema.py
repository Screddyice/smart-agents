"""Initialize a specifically selected empty development database; never migrate production implicitly."""
import argparse
import os
from pathlib import Path
import psycopg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dsn-env',default='SMART_AGENTS_TEST_DSN')
    parser.add_argument('--schema',type=Path,default=Path(__file__).with_name('sql')/'schema.sql')
    args=parser.parse_args()
    dsn=os.environ.get(args.dsn_env)
    if not dsn:parser.error('Selected database environment variable is unset')
    with psycopg.connect(dsn,autocommit=True,prepare_threshold=None) as conn:
        exists=conn.execute("select to_regclass('public.learnings')").fetchone()[0]
        if exists:raise RuntimeError('Database already contains learnings; use an explicit reviewed migration')
        conn.execute(args.schema.read_text())
    print('Initialized selected operational corpus database')


if __name__=='__main__':main()
