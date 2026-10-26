import argparse, torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

def main():
 p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--checkpoint',default='artifacts/best_model.pt');a=p.parse_args();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 ck=torch.load(a.checkpoint,map_location=device,weights_only=False); classes=ck['classes']; model=models.efficientnet_b0(weights=None);model.classifier[1]=nn.Linear(model.classifier[1].in_features,len(classes));model.load_state_dict(ck['model_state_dict']);model.to(device).eval()
 tf=transforms.Compose([transforms.Resize((ck['image_size'],ck['image_size'])),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 with Image.open(a.image).convert('RGB') as im, torch.no_grad(): probs=model(tf(im).unsqueeze(0).to(device)).softmax(1)[0]
 idx=int(probs.argmax());print({'prediction':classes[idx],'confidence':round(float(probs[idx]),4),'probabilities':{c:round(float(probs[i]),4) for i,c in enumerate(classes)}})
if __name__=='__main__':main()
