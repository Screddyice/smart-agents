"""Check repository-relative Markdown links without fetching outside resources."""
from pathlib import Path
import re
root=Path(__file__).resolve().parents[1]
errors=[]
for p in [root/'README.md',root/'CONTRIBUTING.md',root/'SECURITY.md',*root.glob('docs/*.md')]:
    for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
        if '://' in target or target.startswith('#'):continue
        if not (p.parent/target.split('#')[0]).exists():errors.append(f'{p.relative_to(root)}: {target}')
if errors:raise SystemExit('\n'.join(errors))
print('Repository documentation links verified')
