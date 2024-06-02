import random

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torchvision import models
# from fastai.vision.all import *

# configuration for audio processing
n_fft=1024
hop_length=256
target_rate=44100
num_samples=int(target_rate)
au2spec = torchaudio.transforms.MelSpectrogram(sample_rate=target_rate,n_fft=n_fft, hop_length=hop_length, n_mels=256)
ampli2db = torchaudio.transforms.AmplitudeToDB()

def get_x(path, target_rate=target_rate, num_samples=num_samples*2):
    x, rate = torchaudio.load(path)
    if rate != target_rate: 
        x = torchaudio.transforms.Resample(orig_freq=rate, new_freq=target_rate, resampling_method='sinc_interpolation')(x)
    x = x[0] / 32768
    x = x.numpy()
    sample_total = x.shape[0]
    randstart = random.randint(target_rate, sample_total-target_rate*3)
    x = x[randstart:num_samples+randstart]
#     x = fix_length(x, num_samples)
    torch_x = torch.tensor(x)
    spec = au2spec(torch_x)
    spec_db = ampli2db(spec)
    spec_db = spec_db.data.squeeze(0)
    spec_db = spec_db - spec_db.min()
    spec_db = spec_db/spec_db.max()
    return spec_db.unsqueeze(0).unsqueeze(0)

class SingleChannelClassification(nn.Module):
    def __init__(self,):
        super(SingleChannelClassification, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=1)  # Add a 1x1 convolutional layer to convert single channel to 3 channels
        self.backbone = models.resnet18()
        self.fc = nn.Linear(1000, 2)  # Final linear layer for classification

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.backbone(x)
        x = self.fc(x)
        return x


# Create the backbone

def load_model(path='resnet.pth'):
    model = SingleChannelClassification()
    model.eval()
    return model

def get_health_result(model, path, classes=['health', 'unhealth']):
    x = get_x(path)
    p = F.softmax(model(x))[0].tolist()

    return dict(zip(classes, p))
    
if __name__ == '__main__':
    # torch.manual_seed(1)
    # model = load_model()
    # x = get_x('101_1b1_Al_sc_Meditron.wav')
    # print(x)
    # print(model(x))
    model = load_model()
    print(str(get_health_result(model, '101_1b1_Al_sc_Meditron.wav')))