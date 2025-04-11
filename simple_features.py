import warnings
warnings.filterwarnings("ignore")
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
import torch
from tqdm import tqdm
from train.data_loaders import dataset_loader
import numpy as np
import timm
from torchvision.models.efficientnet import efficientnet_v2_l as EfficientNet
from torchvision.models.efficientnet import efficientnet_v2_l as EfficientNet
from torchvision.models.regnet import regnet_y_128gf as RegNet
from train.data_loaders import dataset_loader
# %%
import sys
sys.path.append('./CLIP')  # git clone https://github.com/openai/CLIP 

target_col = 'class_label'  # Column with labels
# %%
torch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cuda" if torch.cuda.is_available() else "cpu"
device

def to_rgb(image):
    return image.convert("RGB")

# %%
def torch_hub_normalization():
    # Normalization for torch hub vision models
    return Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )


# %%
def clip_normalization():
    # SRC https://github.com/openai/CLIP/blob/e5347713f46ab8121aa81e610a68ea1d263b91b7/clip/clip.py#L73
    return Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    )


# %%
# Definel "classic" models from torch-hub
def load_torch_hub_model(model_name):
    # Load model
    model = torch.hub.load('pytorch/vision:v0.6.0',
                           model_name, pretrained=True)

    # Put model in 'eval' mode and sent do device
    model = model.eval().to(device)

    # Check for features network
    if hasattr(model, 'features'):
        features = model.features
    else:
        features = model

    return features, torch_hub_normalization()


def load_mobilenet():
    return load_torch_hub_model('mobilenet_v2')


def load_densenet():
    return load_torch_hub_model('densenet121')


def load_resnet():
    return load_torch_hub_model('resnet101')


def load_resnext():
    return load_torch_hub_model('resnext101_32x8d')


def load_vgg():
    return load_torch_hub_model('vgg16')


def load_crossvit():
    model = timm.create_model('crossvit_15_dagger_240', pretrained=True)
    model = model.eval().to(device)
    return model, torch_hub_normalization()

def load_eva():
    model = timm.create_model('eva_giant_patch14_560.m30m_ft_in22k_in1k', pretrained=True)
    model = model.eval().to(device)
    return model, torch_hub_normalization(), 560

def load_eva02():
    model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True)
    model = model.eval().to(device)
    return model, torch_hub_normalization(), 448

def load_convnext():
    model = timm.create_model('convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384', pretrained=True)
    model = model.eval().to(device)
    return model, torch_hub_normalization(), 384

def load_maxvit():
    model = timm.create_model('maxvit_rmlp_base_rw_384.sw_in12k_ft_in1k', pretrained=True)
    model = model.eval().to(device)
    return model, torch_hub_normalization(), 384

def load_beit():
    model = timm.create_model('beitv2_large_patch16_224.in1k_ft_in22k_in1k', pretrained=True)
    model = model.eval().to(device)
    return model, torch_hub_normalization(), 224

def load_deit():
    model = timm.create_model('deit3_large_patch16_384.fb_in22k_ft_in1k', pretrained=True)
    model = model.eval().to(device)
    return model, torch_hub_normalization(), 384

def load_regnety():
    model = timm.create_model('regnety_320.swag_ft_in1k', pretrained=True)
    model = model.eval().to(device)
    return model, torch_hub_normalization(), 320

def load_regnety2():
    model = timm.create_model('regnety_1280.swag_ft_in1k', pretrained=True)
    model = model.eval().to(device)
    return model, torch_hub_normalization(), 384

def load_swinv2():
    model = timm.create_model('swinv2_large_window12to24_192to384.ms_in22k_ft_in1k', pretrained=True)
    model = model.eval().to(device)
    return model, torch_hub_normalization(), 384

def load_vitlarge():
    model = timm.create_model('vit_large_patch14_clip_336.openai_ft_in12k_in1k', pretrained=True)
    model = model.eval().to(device)
    return model, torch_hub_normalization(), 336

def load_caformer():
    model = timm.create_model('caformer_b36.sail_in22k_ft_in1k_384', pretrained=True)
    model = model.eval().to(device)
    return model, torch_hub_normalization(), 384

def load_coatnet():
    model = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=True)
    model = model.eval().to(device)
    return model, torch_hub_normalization(), 384

def load_vitg14():
    model = timm.create_model('vit_giant_patch14_clip_224.laion2b', pretrained=True)
    model = model.eval().to(device)
    return model, torch_hub_normalization(), 224



def load_eff():
    model = EfficientNet(pretrained=True)
    model = model.eval().to(device)
    if hasattr(model, 'features'):
        features = model.features
    else:
        features = model

    return features, torch_hub_normalization()


def load_RegNet():
    model = RegNet(weights='IMAGENET1K_SWAG_E2E_V1')
    model = model.eval().to(device)
    if hasattr(model, 'features'):
        features = model.features
    else:
        features = model

    return features, torch_hub_normalization()
# %%
# Define CLIP models (ViT-B and RN50)
def load_clip_vit_b():
    model, _ = clip.load("ViT-B/32", device=device)

    return model.encode_image, clip_normalization(), 224


def load_clip_rn50():
    model, _ = clip.load("RN50", device=device)

    return model.encode_image, clip_normalization(), 224

# %%
# Dataset loader
class ImagesDataset(Dataset):
    def __init__(self, df, preprocess, input_resolution):
        super().__init__()
        self.df = df
        self.preprocess = preprocess
        self.empty_image = torch.zeros(3, input_resolution, input_resolution)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        try:
            if 'Filename' in row:
                image = self.preprocess(Image.open(row['Filename']))
            else:
                image = self.preprocess(Image.fromarray(row['Image']))    
        except:
            print("error")
            image = self.empty_image

        return image, row[target_col]


# %%
# Define model loaders
MODELS_LOADERS = {
    # 'mobilenet': load_mobilenet,
    #'densenet': load_densenet,
    #'efficient_net': load_eff,
    #'nfnet_f4' : load_NFNet,
    #'regnet' : load_RegNet,
    # 'resnet': load_resnet,
    # 'resnext': load_resnext,
    # 'vgg': load_vgg,
    #'eva_giant' : load_eva,
    #'regnety1280' : load_regnety2,
    #'eva2' : load_eva2,
    'vitg14' : load_vitg14
    #'regnety': load_regnety,
    #'eva02': load_eva02,
    #'convnext': load_convnext,
    #'maxvit': load_maxvit,
    #'beit': load_beit,
    #'deit3': load_deit,
    #'swinv2': load_swinv2,
    #'vit_large': load_vitlarge,
    #'caformer': load_caformer,
    #'coatnet': load_coatnet,
    #'clip_vit_b': load_clip_vit_b,
    #'clip_rn50': load_clip_rn50
}


# %%
# Main function to generate features
def generate_features(model_loader):
    # Create model and image normalization
    model, image_normalization, input_resolution = model_loader()

    # General transformation applied to all models
    preprocess_image = Compose(
        [
            Resize(input_resolution, interpolation=Image.BICUBIC),
            CenterCrop(input_resolution),
            to_rgb,
            ToTensor(),
        ]
    )

    preprocess = Compose([preprocess_image, image_normalization])

    # Create DataLoader
    ds = ImagesDataset(df_all, preprocess, input_resolution)
    dl = DataLoader(ds, batch_size=32, shuffle=False,
                    pin_memory=True)

    # Sample one output from model just to check output_dim
    x = torch.zeros(1, 3, input_resolution, input_resolution, device=device)
    with torch.no_grad():
        x_out = model(x)
    output_dim = x_out.shape[1]

    # Features data
    X = np.empty((len(ds), output_dim), dtype=np.float32)
    y = np.empty(len(ds), dtype=np.int32)

    # Begin feature generation
    i = 0
    for images, cls in tqdm(dl):
        n_batch = len(images)

        with torch.no_grad():
            emb_images = model(images.to(device))
            if emb_images.ndim == 4:
                emb_images = emb_images.reshape(
                    n_batch, output_dim, -1).mean(-1)
            emb_images = emb_images.cpu().float().numpy()

        # Save normalized features
        X[i:i+n_batch] = emb_images / \
            np.linalg.norm(emb_images, axis=1, keepdims=True)
        y[i:i+n_batch] = cls

        i += n_batch

    del model, image_normalization, ds, dl

    return X, y, output_dim

dataset_name = 'cifar10'
df_all, LABELS_MAP, _ = dataset_loader(dataset_name)

for i, (model_name, model_loader) in enumerate(MODELS_LOADERS.items(), 1):
    print(f'Extracting features with {model_name}...')

    X, y, _ = generate_features(model_loader)

    np.save(f"D:/features_models/X_" + str(dataset_name) + "_" + str(model_name) + ".npy", X)
    np.save(f"y_" + str(dataset_name) + ".npy", y)
    
