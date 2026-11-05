"""Smart Agents operator commands. No command is run during package installation."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    weekly=sub.add_parser('weekly',help='prepare or explicitly train the local cross-encoder')
    weekly.add_argument('--triples',type=Path,default=os.environ.get('SMART_AGENTS_TRIPLES'))
    weekly.add_argument('--state',type=Path,default=Path(os.environ.get('SMART_AGENTS_STATE_DIR',Path.home()/'.local/state/smart-agents')))
    weekly.add_argument('--dry-run',action='store_true')
    weekly.add_argument('--build-count',type=int,default=0,help='explicitly prepare this many new labeled examples before training')
    weekly.add_argument('--base',default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    for name in ('record','feedback','review-feedback'):
        command=sub.add_parser(name,help='explicit trusted administrative write from a JSON file')
        command.add_argument('file',type=Path)
    sub.add_parser('mcp',help='serve the read-only MCP over stdio')
    sub.add_parser('coverage',help='show capture boundaries')
    args=parser.parse_args()
    if args.command=='weekly':
        if not args.triples:parser.error('--triples or SMART_AGENTS_TRIPLES is required')
        if args.build_count:
            if args.dry_run:parser.error('Dry run does not call external labelers; build examples separately')
            if not 1<=args.build_count<=1000:parser.error('build-count must be between 1 and 1000')
            import subprocess,sys
            subprocess.run([sys.executable,'-m','smart_agents.rerank.triples','--n',str(args.build_count),'--out',str(args.triples)],check=True)
        from .weekly import run
        print(run(triples=args.triples,state=args.state,dry_run=args.dry_run,base=args.base))
    elif args.command in ('record','feedback','review-feedback'):
        from .admin import post
        value=json.loads(args.file.read_text())
        if args.command=='feedback':
            from .feedback import Feedback
            value=Feedback.model_validate(value).model_dump()
        if args.command=='review-feedback':
            from .feedback import FeedbackReview
            feedback_id=value.pop('feedback_id')
            if type(feedback_id) is not int or feedback_id<1:parser.error('Invalid feedback_id')
            value=FeedbackReview.model_validate(value).model_dump()
            path=f'/feedback/{feedback_id}/review'
        else:path='/corrections' if args.command=='record' else '/feedback'
        print(json.dumps(post(path,value)))
    elif args.command=='mcp':
        from .mcp import main as serve
        serve()
    else:
        print('cmem: distilled observations and summaries only; raw prompts excluded.\nExplicit record: trusted correction/discovery evidence.\nOther hosts: capture is verified per profile/source; MCP availability alone does not prove capture.')


if __name__=='__main__':main()
