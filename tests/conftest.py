import os
# Deterministic unit suite never loads a model or uses a production database.
os.environ['RERANK_ENABLED']='0'
os.environ.setdefault('RDS_URL','postgresql://test:test@localhost/test')
