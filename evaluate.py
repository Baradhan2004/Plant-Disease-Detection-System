"""Evaluate a saved EfficientNet-B0 checkpoint on a held-out ImageFolder."""
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); ap.add_argument('--checkpoint',default='artifacts/best_model.pt'); ap.add_argument('--batch-size',type=int,default=32); args=ap.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ckpt=torch.load(args.checkpoint,map_location=device,weights_only=False)
    image_size=ckpt.get('image_size',224)
    tfm=transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    ds=datasets.ImageFolder(args.data_dir,transform=tfm); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=0)
    model=models.efficientnet_b0(weights=None); model.classifier[1]=torch.nn.Linear(model.classifier[1].in_features,len(ds.classes)); model.load_state_dict(ckpt['model_state_dict']); model.to(device).eval()
    ys=[]; ps=[]
    with torch.inference_mode():
        for x,y in loader:
            pred=model(x.to(device)).argmax(1).cpu(); ys.extend(y.tolist()); ps.extend(pred.tolist())
    print(json.dumps({'samples':len(ys),'accuracy':accuracy_score(ys,ps),'macro_f1':f1_score(ys,ps,average='macro',zero_division=0),'classes':ds.classes,'classification_report':classification_report(ys,ps,target_names=ds.classes,zero_division=0,output_dict=True),'confusion_matrix':confusion_matrix(ys,ps).tolist()},indent=2))
if __name__=='__main__': main()
