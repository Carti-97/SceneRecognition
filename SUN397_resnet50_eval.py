import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoConfig, CLIPModel, CLIPTokenizer
import pandas as pd
import os
from PIL import Image
import numpy as np
import random
import math
from tqdm import tqdm
import timm
from torchvision.transforms import v2
from torchvision import transforms
from torch.utils.data import default_collate
import torchvision.models as models
from SASNET.RGBBranch import RGBBranch
#from torch.utils.tensorboard import SummaryWriter

    
device = 'cuda:1'


def set_seed(seed=42):
    """Set the seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#set_seed(42) 


class CLIPDataset(Dataset):
    def __init__(self, image_dir, tags_csv, train=True):
        self.image_dir = image_dir
        self.tags_df = pd.read_csv(tags_csv)
        
        # CSV 파일에서 이미지 경로와 태그를 추출
        self.image_paths = self.tags_df['image_path'].tolist()
        self.image_paths = [path.replace('/home/hdd2/', '/home/pilab/') if '/home/hdd2/' in path else 
                    path.replace('/home/pi/', '/home/pilab/') if '/home/pi/' in path else 
                    path for path in self.image_paths]
        self.tags = self.tags_df['tag'].tolist()
     
        
        # 레이블은 이미지 경로에서 폴더명을 사용
        self.labels = [path.split('/')[-2] for path in self.image_paths]
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}

        if train:
            self.transform = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomResizedCrop(size=(224, 224)),
                v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                v2.RandomAdjustSharpness(sharpness_factor=2),
                #v2.RandomApply([v2.GaussianNoise()], p=0.5),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
                v2.Resize((224, 224)),
                v2.ToTensor(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        # 검증용 변환
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.TenCrop(size=(224, 224)), # 이미지 크기를 조정
                transforms.Lambda(lambda crops: torch.stack([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                    transforms.ToTensor()(crop)
                ) for crop in crops
            ]))
        ])

         
    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        label = self.labels[idx]

        image = Image.open(image_path).convert('RGB')
        
        transformed_crops = self.transform(image)  # Should return a stack of tensors

        label_index = self.label_to_index[label]
        return transformed_crops, label_index



#model = timm.create_model("hf_hub:timm/resnet50.a1_in1k", pretrained=True)

#model = models.resnet50(pretrained=False)

model = RGBBranch(arch='ResNet-50', scene_classes=397)
model.to(device)

checkpoint = torch.load('/home/pilab/chbae/scene_recognize/RGB_ResNet50_SUN.pth.tar')
print(checkpoint.keys())
model.load_state_dict(checkpoint['state_dict'])

# 분류를 위한 추가 레이어 정의
class ClassifierWithTransformer(nn.Module):
    def __init__(self, model, num_classes):
        super(ClassifierWithTransformer, self).__init__()
        
        """
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        text_embedding = self.clip.get_text_features(input_ids=tag, attention_mask=attention_mask)

        # 텍스트 관련 파라미터는 고정
        for param in self.clip.text_model.parameters():
            param.requires_grad = False
        """
        
        self.model = model

        for param in self.model.parameters():
            param.requires_grad=False

        self.classifier = nn.Linear(1000, num_classes)  # 예시로 768은 모델의 출력 차원, num_classes는 분류하고자 하는 클래스 수

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        
        logits = self.classifier(outputs)
        return logits


    
# 하이퍼파라미터 설정
num_classes = 67
learning_rate = 0.0001
batch_size = 16
epochs = 200

# 데이터셋 및 데이터로더 생성 
images = '/home/pilab/chbae/SUN397_split/train/'  # 이미지 데이터
tags = '/home/pilab/chbae/SUN397_split/train_tags.csv'  # 태그 데이터
images_val = '/home/pilab/chbae/SUN397_split/val/'
tags_val = '/home/pilab/chbae/SUN397_split/val_tags.csv'


train_dataset = CLIPDataset(images, tags, train=True)
val_dataset = CLIPDataset(images_val, tags_val, train=False)


# 모델 초기화 및 학습
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

#model = ClassifierWithTransformer(model, num_classes).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

#writer = SummaryWriter('/home/pi/chbae/scene_recognize/logs/resnet50')

best_val_accuracy = 0.0
best_epoch = 0
best_model_state = None
"""
for epoch in range(epochs):
    model.train()
    correct_train = 0
    total_train = 0
    train_loss = 0.0
    
    for inputs_image, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch'):
        
        inputs_image = inputs_image.to(device)
        
        labels = labels.to(device=device)

        # Forward pass
        outputs = model(inputs_image)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        
        optimizer.step()

        
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

        train_loss += loss.item()
        

    #train_accuracy = correct_train / total_train
    #train_loss /= len(train_loader)


    #writer.add_scalar('Loss/train', train_loss, epoch)
    #writer.add_scalar('Accuracy/train', train_accuracy, epoch)

    # Validation phase
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs_image, labels in val_loader:
            inputs_image = inputs_image.to(device)
            
            labels = labels.to(dtype=torch.long, device=device)

            outputs = model(inputs_image)
            _, predicted_val = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted_val == labels).sum().item()

            val_loss += loss.item()

    val_accuracy = correct_val / total_val
    #val_loss /= len(val_loader)

    # 검증 단계에서의 손실과 정확도를 TensorBoard에 기록
    #writer.add_scalar('Loss/validation', val_loss, epoch)
    #writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        best_model_state = model.state_dict().copy()

    print(f'Epoch {epoch+1}, Validation Accuracy: {val_accuracy:.4f}')

print(f'best_val_accuracy: {best_val_accuracy:.4f}, best_epoch: {best_epoch}')

#writer.close()
"""    

model.eval()
correct_val = 0
total_val = 0
val_loss = 0.0

with torch.no_grad():
    for inputs_image, labels in val_loader:
        #print(inputs_image.size())
        bs, crops, c, h, w = inputs_image.size()
        
        # Merge the batch_size and crops dimensions
        inputs_image = inputs_image.view(-1, c, h, w).to(device)
        
        labels = labels.to(dtype=torch.long, device=device)

        outputs = model(inputs_image)
        outputs = outputs.view(bs, crops, -1)
        
        # Calculate the predictions for each crop
        _, preds = torch.max(outputs, 2)
        
        # Select the most common prediction for each image
        predicted_val = torch.mode(preds, 1).values

        total_val += labels.size(0)
        correct_val += (predicted_val == labels).sum().item()


        #val_loss += loss.item()

val_accuracy = correct_val / total_val

print(f'Validation Accuracy: {val_accuracy:.4f}')
