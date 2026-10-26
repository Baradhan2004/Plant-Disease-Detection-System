import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='data'); ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--image-size',type=int,default=224); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--workers',type=int,default=0); args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    root=Path(args.data_dir); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean,std=[.485,.456,.406],[.229,.224,.225]
    tf={'train':transforms.Compose([transforms.Resize((args.image_size,args.image_size)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(12),transforms.ToTensor(),transforms.Normalize(mean,std)]),'val':transforms.Compose([transforms.Resize((args.image_size,args.image_size)),transforms.ToTensor(),transforms.Normalize(mean,std)])}
    ds={s:datasets.ImageFolder(root/s,tf[s]) for s in ('train','val')}
    if ds['train'].classes != ds['val'].classes: raise ValueError('Train/val class folders must match exactly')
    loaders={s:DataLoader(ds[s],batch_size=args.batch_size,shuffle=s=='train',num_workers=args.workers) for s in ds}
    weights=models.EfficientNet_B0_Weights.DEFAULT; model=models.efficientnet_b0(weights=weights); model.classifier[1]=nn.Linear(model.classifier[1].in_features,len(ds['train'].classes)); model.to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4); loss_fn=nn.CrossEntropyLoss(); best=-1; out=Path('artifacts');out.mkdir(exist_ok=True)
    for epoch in range(args.epochs):
      for phase in ('train','val'):
        model.train(phase=='train'); y,preds=[],[]; total_loss=0
        for x,yb in loaders[phase]:
          x,yb=x.to(device),yb.to(device); opt.zero_grad(set_to_none=True)
          with torch.set_grad_enabled(phase=='train'):
            logits=model(x); loss=loss_fn(logits,yb)
            if phase=='train': loss.backward();opt.step()
          total_loss+=loss.item()*len(yb); y.extend(yb.cpu().tolist());preds.extend(logits.argmax(1).cpu().tolist())
        acc=float(np.mean(np.array(y)==np.array(preds))); macro=float(f1_score(y,preds,average='macro',zero_division=0))
        print(f'epoch={epoch+1} split={phase} loss={total_loss/len(ds[phase]):.4f} accuracy={acc:.4f} macro_f1={macro:.4f}')
        if phase=='val' and acc>best:
          best=acc; torch.save({'model_state_dict':model.state_dict(),'classes':ds['train'].classes,'image_size':args.image_size},out/'best_model.pt')
          (out/'classes.json').write_text(json.dumps(ds['train'].classes, indent=2), encoding='utf-8')
          (out/'validation_report.json').write_text(json.dumps({'epoch':epoch+1,'accuracy':acc,'macro_f1':macro,'classification_report':classification_report(y,preds,target_names=ds['train'].classes,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y,preds).tolist()},indent=2))
    print('Best validation accuracy:',best,'(not a test-set score)')
if __name__=='__main__': main()
