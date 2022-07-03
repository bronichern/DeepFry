import torch
import torch.nn as nn
from utils import LambdaLayer


class ClassifierPred(nn.Module):
    def __init__(self, input_size=512, hidden_size=256, classes=1, dropout=0):
        super(ClassifierPred, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.classes = classes
        self.fc2 = nn.Linear(hidden_size, classes)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        probs = self.sig(self.fc2(x))
        return probs,self.fc2(x)
    
    def get_config(self):
        config = {}
        config.update({"input_size": self.input_size})
        config.update({"hidden_size": self.hidden_size})
        config.update({"classes": self.classes})
        return config

class ClassifierPred1Layer(nn.Module):
    def __init__(self, input_size=512, classes=100, dropout=0):
        super(ClassifierPred1Layer, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.fc1 = nn.Linear(input_size, classes)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        probs = self.sig(self.fc1(x))
        return probs,self.fc1(x)

    def get_config(self):
        config = {}
        config.update({"input_size": self.input_size})
        config.update({"classes": self.classes})
        return config

class CnnRaw(nn.Module):	
    def __init__(self, enc_out_div=1, input_size=512, channels=256, dropout=0.2, hidden = 128):	
        super(CnnRaw, self).__init__()	
        self.enc = nn.Sequential(	
            nn.Conv1d(1, channels, kernel_size=512, stride=5, padding=256, bias=False),	
            nn.BatchNorm1d(channels),	
            nn.LeakyReLU(),	
            nn.Dropout(p=dropout),	
            nn.Conv1d(channels, 128, kernel_size=64, stride=4, padding=31, bias=False),	
            nn.BatchNorm1d(128),	
            nn.LeakyReLU(),	
            nn.Dropout(p=dropout),	
            nn.Conv1d(128, hidden, kernel_size=32, stride=2, padding=15, bias=False),	
            nn.BatchNorm1d(hidden),	
            nn.LeakyReLU(),	
            nn.Dropout(p=dropout),	
            nn.Conv1d(hidden, hidden, kernel_size=8, stride=2, padding=4, bias=False),	
            nn.BatchNorm1d(hidden),	
            nn.LeakyReLU(),	
            nn.Dropout(p=dropout),	
            nn.Conv1d(hidden, int(input_size//enc_out_div), kernel_size=4, stride=1, padding=1, bias=False),	
            nn.BatchNorm1d(int(input_size//enc_out_div)),	
            nn.LeakyReLU(),	
            LambdaLayer(lambda x: x.transpose(1,2)),	
        )	
        self.input_size = input_size	
        self.channels = channels	

    def forward(self, raw_input):

        enc_input = self.enc(raw_input)
        return enc_input.squeeze()

    def get_config(self):
        config = {}
        config.update({"input_size": self.input_size})
        config.update({"channels": self.channels})
        return config

class EncoderClassifier(nn.Module):
    def __init__(self, enc_out_div=1, input_size=512, channels=256, dropout=0.5, hidden_size=256, enc_hidden=128, classes=1):
        super(EncoderClassifier, self).__init__()
        self.input_size = input_size
        self.channels = channels
        self.hidden_size = hidden_size

        self.enc = CnnRaw(enc_out_div=enc_out_div, input_size=input_size, channels=channels, dropout=dropout, hidden = enc_hidden)
        self.classifier = ClassifierPred(input_size=int(input_size//enc_out_div), hidden_size=hidden_size, classes=classes)


    def forward(self, x):

        enc_vec = self.enc(x)
        output,logits = self.classifier(enc_vec)
        return logits, output


    def get_config(self):
        config = {}
        config.update(self.enc.get_config())
        config.update(self.classifier.get_config())
        return config

    @classmethod
    def from_config(cls, config_dict):
        return cls(**config_dict)

def build_model(config_dict):
    if "enc_out_div" not in config_dict:
        config_dict.enc_out_div = 1
    if "enc_hidden" not in config_dict:
        config_dict.enc_hidden = 128
    print("building backbone:")
    print(f"input_size:{config_dict.input_size}, channels:{config_dict.channels}, dropout:{config_dict.dropout}, hidden_size:{config_dict.hidden_size}, classes:{config_dict.classes},enc_out_div:{config_dict.enc_out_div}, enc_hidden:{config_dict.enc_hidden}")
    model = EncoderClassifier.from_config({"input_size":config_dict.input_size, "channels":config_dict.channels, 
    "dropout":config_dict.dropout, "hidden_size":config_dict.hidden_size, "classes":config_dict.classes,"enc_out_div":config_dict.enc_out_div,"enc_hidden":config_dict.enc_hidden })
    print(model)
    return model

def load_model(path, args, device='cpu', strict=True):
    checkpoint = torch.load(path, map_location='cpu')
    params = checkpoint['model_params']
    running_params = checkpoint.get('running_params',"")
    if 'enc_out_div' not in params:
        params['enc_out_div'] = 1
    if 'enc_hidden' not in params:
        params['enc_hidden'] = args.enc_hidden if 'enc_hidden' in args else 128
    print(f"loading model with params :{params}")
    model = EncoderClassifier.from_config(params)
    model.load_state_dict(checkpoint['net'],strict=strict)
    return model, running_params