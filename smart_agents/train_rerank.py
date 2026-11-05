"""Explicit local cross-encoder fine-tune; select checkpoints using validation only."""
import argparse
import json
import math
from pathlib import Path
from .training import write_new_json


def auc(scores, labels):
    pos=[s for s,l in zip(scores,labels) if l==1]
    neg=[s for s,l in zip(scores,labels) if l==0]
    if not pos or not neg:raise ValueError('Validation requires both positive and negative examples')
    return (sum(p>n for p in pos for n in neg)+.5*sum(p==n for p in pos for n in neg))/(len(pos)*len(neg))


def pairs(rows):
    result=[]
    for row in rows:
        relevant=set(row['relevant_ids'])
        for c in row['candidates']:
            result.append({'task':row['task'],'lesson':c['lesson'],'label':int(c['id'] in relevant)})
    return result


def train(prepared, output, *, base, epochs=2, batch_size=16, lr=1e-5, seed=0):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from .rerank.dataset import balance
    data=json.loads(Path(prepared).read_text())
    train_pairs=balance(pairs(data['train']),max_neg_per_pos=8,seed=seed)
    val=pairs(data['validation'])
    if not train_pairs or {x['label'] for x in val}!={0,1}:
        raise ValueError('Independent train and mixed-label validation groups required')
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(base)
    model=AutoModelForSequenceClassification.from_pretrained(base).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    loss_fn=torch.nn.BCEWithLogitsLoss()
    def encode(batch):
        return tok([b['task'] for b in batch],[b['lesson'] for b in batch],padding=True,truncation=True,max_length=384,return_tensors='pt').to(device)
    loader=DataLoader(train_pairs,batch_size=batch_size,shuffle=True,collate_fn=lambda x:x)
    best=None;history=[]
    for epoch in range(1,epochs+1):
        model.train();total=0.
        for batch in loader:
            logits=model(**encode(batch)).logits.squeeze(-1)
            labels=torch.tensor([float(b['label']) for b in batch],device=device)
            loss=loss_fn(logits,labels)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad();loss.backward();optimizer.step();total+=loss.item()*len(batch)
        model.eval();scores=[]
        with torch.no_grad():
            for i in range(0,len(val),64):scores+=model(**encode(val[i:i+64])).logits.squeeze(-1).float().cpu().tolist()
        val_auc=auc(scores,[b['label'] for b in val])
        if not math.isfinite(val_auc):raise RuntimeError('Nonfinite validation AUC')
        checkpoint=output/f'epoch-{epoch}';checkpoint.mkdir()
        model.save_pretrained(checkpoint);tok.save_pretrained(checkpoint)
        record={'epoch':epoch,'train_loss':total/len(train_pairs),'validation_auc':val_auc}
        history.append(record);print(json.dumps(record),flush=True)
        if best is None or val_auc>best[0]:best=(val_auc,epoch)
    (output/'model').symlink_to(f'epoch-{best[1]}',target_is_directory=True)
    write_new_json(output/'metrics.json',{'base':base,'best_epoch':best[1],'history':history,'device':device,'seed':seed})
    return output/'model'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--base',default='cross-encoder/ms-marco-MiniLM-L-6-v2')
    ap.add_argument('--epochs',type=int,default=2)
    args=ap.parse_args()
    if not 1<=args.epochs<=20:ap.error('epochs must be between 1 and 20')
    train(args.prepared,args.output,base=args.base,epochs=args.epochs)


if __name__=='__main__':main()
