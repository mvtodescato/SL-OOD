import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
import torch
from tqdm import tqdm
import numpy as np
from train.data_loaders import dataset_loader
import faiss
from statistics import mean, stdev
from sklearn.metrics import roc_auc_score

import getopt, sys
from scipy.io import loadmat
sys.path.append('./CLIP') 


target_col = 'class_label'  # Default column with labels
input_resolution = 224    # Default input resolution


# %%
torch_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %%
def to_rgb(image):
    return image.convert("RGB")


# General transformation applied to all models
preprocess_image = Compose(
    [
        Resize(input_resolution, interpolation=Image.BICUBIC),
        CenterCrop(input_resolution),
        to_rgb,
        ToTensor(),
    ]
)


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
            image = self.empty_image

        return image, row[target_col]






def map_indices(original_list, reordered_list):
    if len(original_list) != len(reordered_list):
        raise ValueError("Both lists must have the same length")
    
    mapping = {}
    for original_index, class_name in enumerate(original_list):
        new_index = reordered_list.index(class_name)
        mapping[original_index] = new_index
    
    return mapping

def translate_indices(mapping, original_indices):
    return np.array([mapping[index] for index in original_indices])

def translate_dataframe(df, mapping, column_name):
    df[column_name] = df[column_name].apply(lambda x: mapping[x])
    return df

def add_vector_to_index(embedding, index):
    #convert embedding to numpy
    vector = embedding.detach().cpu().numpy()
    #Convert to float32 numpy
    vector = np.float32(vector)
    #Normalize vector
    faiss.normalize_L2(vector)
    #Add to index
    index.add(vector)

DATASETS = {
    'cifar100' : 20,
    'cifar10' : 6,
    'cars' : 120,
    'geo' : 30,
    'caltech' : 21,
    'caltech256' : 52,
}

def ood_score(mult,dataset):
    #Set the dataset, load the features and map the splits
    dataset_name = dataset
    df_all, LABELS_MAP, SPLITS = dataset_loader(dataset_name)

    SPLIT_SIZE = DATASETS[dataset_name]

    class BaseNeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(1024, SPLIT_SIZE)     
            )

        def forward(self, x):
            return self.network(x)


    with open('D:/features_models/X_' + dataset_name + '_vitg14.npy','rb') as f:
        X = np.load(f)

    with open('y_' + dataset_name + '.npy','rb') as f:      
        y = np.load(f)

    original_LABELS_MAP = LABELS_MAP
    original_y = y
    original_df_all = df_all.copy()

    auc_list_sum = []

    for split in SPLITS:
        index_mapping = map_indices(original_LABELS_MAP, split)
        print(index_mapping)
        LABELS_MAP = split
        y = translate_indices(index_mapping, original_y)
        df_aux = original_df_all.copy()
        df_all = translate_dataframe(df_aux, index_mapping, 'class_label')

        selected_indices = []

        #Random selection of 5 images of each seen class (5-shot)
        classes = list(range(SPLIT_SIZE))
        all_indices = list(range(len(df_all)))

        for class_ in classes:
            class_indices = []
            
            class_indices = df_all[df_all['class_label'] == class_].index.tolist()
            
            if dataset_name == 'geo':
                class_indices_row = [df_all.index.get_loc(key) for key in class_indices]
            
                for i in range(len(class_indices_row)):
                    if type(class_indices_row[i]) == slice:
                        class_indices_row[i] = all_indices[class_indices_row[i]][0]
                random_indices = np.random.choice(class_indices_row, size=5, replace=False)
            else:
                random_indices = np.random.choice(class_indices, size=5, replace=False)

            selected_indices.extend(random_indices)

        few_shot_indices = selected_indices

        train_X = X[few_shot_indices]
        train_y = y[few_shot_indices]

        fold_model = BaseNeuralNetwork()
        fold_model.to(torch_device)

        print("Classifier")
        print(fold_model)
        print()

        fold_criterion = nn.CrossEntropyLoss()

        fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=0.001)

        EPOCHS = 100
        print("Loss", fold_criterion)
        print("Optimizar", fold_optimizer)
        print()

        #Initial training
        print("Start training ...")
        for epoch in range(EPOCHS):
                    running_loss = 0.0

                    train_loader = DataLoader(list(zip(train_X, train_y)), batch_size=16, shuffle=True,
                                num_workers=0, pin_memory=True)

                    for i, data in enumerate(train_loader, 0):

                        inputs, label_index = data

                        multilabel_values = np.zeros((len(label_index),SPLIT_SIZE)).astype(float)

                        for k, idx in enumerate(label_index):
                            multilabel_values[k][idx] = 1.0


                        tensor_multilabel_values = torch.from_numpy(multilabel_values).to(torch_device)

                        fold_optimizer.zero_grad()

                        outputs = fold_model(inputs.to(torch_device))
                        pred = outputs.cpu().argmax()

                        fold_loss = fold_criterion(outputs, tensor_multilabel_values.float())
                        fold_loss.backward()
                        fold_optimizer.step()

                        running_loss += fold_loss.item()
                        
                        if i == len(train_loader) - 1:  
                            print('[%d, %5d] Train loss: %.5f' %
                                (epoch + 1, i + 1, running_loss / len(train_loader)))
                            running_loss = 0.0


        image_normalization = torch_hub_normalization()
        preprocess = Compose([preprocess_image, image_normalization])
        
        df_all = df_all.reset_index(drop=True)
        df_new = df_all.drop(few_shot_indices)
        df_new = df_new.reset_index(drop=True)
        new_indices = [index for index in all_indices if index not in selected_indices]
        new_X = X[new_indices]

        ds = ImagesDataset(df_all, preprocess, input_resolution)

        model, _ = clip.load("ViT-B/32", device=device)
        classes = [(f"a photo of a {c}") for c in LABELS_MAP[:SPLIT_SIZE]]

        try:
            with open('X_clip_' + dataset_name + '.npy','rb') as f:
                X_clip = np.load(f)
        except:
            X_clip = np.empty((len(ds), 512), dtype=np.float32)
            i = 0
            for images, class_name in tqdm(ds):
                    image = images.unsqueeze(0).to(device)

                    with torch.no_grad():
                        image_features = model.encode_image(image)
                    image_features = image_features.detach().cpu().numpy()

                    X_clip[i] = np.float32(image_features)

                    i += 1
            np.save(f'X_clip_' + dataset_name + '.npy', X_clip)

        X_index = X_clip[new_indices]
        index = faiss.IndexFlatL2(512)
        for embedding in X_index:
            
            embedding = torch.from_numpy(embedding)
            embedding = embedding.to(device)
            embedding = embedding.unsqueeze(0)

            add_vector_to_index(embedding,index)

        #Generate the ranking for each seen class
        top5list = []
        for id_class in range(len(classes)):
            text = clip.tokenize(classes[id_class])
            with torch.no_grad():
                text_features = model.encode_text(text.cuda())

            text_np = text_features.detach().cpu().numpy()
            text_np = np.float32(text_np)

            distances, indices = index.search(text_np, 100)

            top5list.append(indices[0])

            for i,v in enumerate(indices[0]):
                sim = (1/(1+distances[0][i])*100)

        
        first_loss = 100
        #Start the self-training
        for i in range(20):
            start = i*5
            selected_indices = []
            train_y = []
            cont_class = 0
            
            #Selecting the next top-5 images of each ranking and testing in the classifier
            for top5 in top5list:
                for indeximg in range(start,start+5):
                    features = torch.from_numpy(new_X[top5[indeximg]])
                    preds = fold_model(features.to(torch_device))
                    pred_index = preds.cpu().argmax()
                    if pred_index == cont_class:
                        selected_indices.append(top5[indeximg])
                        train_y.append(cont_class)
                cont_class += 1


            print("Selected images")
            print(selected_indices)
            print(len(selected_indices))

            train_X = new_X[selected_indices]

            break_train = 0
            EPOCHS = 20
            print("Loss", fold_criterion)
            print("Optimizar", fold_optimizer)
            print()
            print("Start training ...")

            #Tune the classifier with the selected images
            for epoch in range(EPOCHS):  
                        if break_train == 1:
                            break
                        running_loss = 0.0

                        train_loader = DataLoader(list(zip(train_X, train_y)), batch_size=16, shuffle=True,
                                    num_workers=0, pin_memory=True)

                        for i, data in enumerate(train_loader, 0):
                            inputs, label_index = data

                            multilabel_values = np.zeros((len(label_index),SPLIT_SIZE)).astype(float)

                            for k, idx in enumerate(label_index):
                                multilabel_values[k][idx] = 1.0


                            tensor_multilabel_values = torch.from_numpy(multilabel_values).to(torch_device)

                            fold_optimizer.zero_grad()

                            outputs = fold_model(inputs.to(torch_device))
                            pred = outputs.cpu().argmax()

                            fold_loss = fold_criterion(outputs, tensor_multilabel_values.float())

                            fold_loss.backward()
                            fold_optimizer.step()


                            running_loss += fold_loss.item()
                            
                            if i == len(train_loader) - 1:   
                                print('[%d, %5d] Train loss: %.5f' %
                                    (epoch + 1, i + 1, running_loss / len(train_loader)))
                                if epoch == 0:
                                    if (running_loss / len(train_loader)) > first_loss:
                                        break_train = 1
                                    else:
                                        first_loss = running_loss / len(train_loader)
                                if (running_loss / len(train_loader)) < 0.1:
                                    break_train = 1
                                running_loss = 0.0

            test_X = X[few_shot_indices]
            test_y = y[few_shot_indices]

            std_list = []
            predvalues = []
            difvalue = []
            #Extract the statistics from the seed
            for x_item, y_item in list(zip(test_X, test_y)):

                    item_input = torch.from_numpy(x_item).to(torch_device)
                    preds = fold_model(item_input)

                    std_list.append(torch.std(preds).item())
                    
                    pred_index = preds.cpu().argmax()
                    value , _ = preds.topk(2)
                    predvalues.append(value[0].item())
                    difvalue.append(value[0].item() - value[1].item())

            if break_train == 1:
                break


        #Calculate the thresholds
        print("mean top1", mean(predvalues))
        print("stdev top1", stdev(predvalues))
        threshold = mean(predvalues) - stdev(predvalues) * mult
        print("threshold 1", threshold)
        print("mean dif top1 top2", mean(difvalue))
        print("stdev dif", stdev(difvalue))
        threshold2 = mean(difvalue) - stdev(difvalue) * mult
        print("threshold 2", threshold2)
        print("stdev output values", mean(std_list))
        print("stdev stdev", stdev(std_list))
        threshold3 = mean(std_list) - stdev(std_list) * mult
        print("threshold 3", threshold3)


        
        X_test = X
        y_test = y

        flag1 = 0
        flag2 = 0
        flag3 = 0
        true_y = []
        predict_y = []
        #Test and calculate the OOD score of the dataset images
        for x_item, y_item in list(zip(X_test, y_test)):
                flag1 = 0
                flag2 = 0
                flag3 = 0

                item_input = torch.from_numpy(x_item).to(torch_device)
                preds = fold_model(item_input)
                
                pred_index = preds.cpu().argmax()
                value , _ = preds.topk(2)
                
                if (y_item>=SPLIT_SIZE):
                    true_y.append(1)
                else:
                    true_y.append(0)

                if value[0] < threshold:
                    flag1 = 1
                
                if (value[0] - value[1]) < threshold2:
                    flag2 = 1

                if torch.std(preds).item() < threshold3:
                    flag3 = 1

                if (flag1) and (flag2) and (flag3):
                    predict_y.append(1)
                elif ((flag1) and (flag2)) or ((flag1) and (flag3)) or ((flag3) and (flag2)):
                    predict_y.append(0.66)
                elif (flag1 or flag2 or flag3):
                    predict_y.append(0.33)
                else:
                    predict_y.append(0)

        auc_sum = roc_auc_score(np.array(true_y), np.array(predict_y))
        auc_list_sum.append(auc_sum)
        print("roc auc", auc_sum)

    print(np.mean(auc_list_sum), np.std(auc_list_sum), auc_list_sum[0],auc_list_sum[1],auc_list_sum[2],auc_list_sum[3],auc_list_sum[4])

    with open("results_ood_" + str(dataset_name) + ".txt", "a") as text_file:
        text_file.write(str(dataset_name) + ' ' + str(mult) + ' ' + str(np.mean(auc_list_sum)) + ' ' + str(np.std(auc_list_sum)) + "\n")


def main():
    argumentList = sys.argv[1:]
    # Options
    options = "hM:D:"
    # Long options
    long_options = ["Help", "Mult=", "Dataset="]
    mult = 0
    dataset = ''
    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        
        # checking each argument
        for currentArgument, currentValue in arguments:
    
            if currentArgument in ("-h", "--help"):
                print("Confidence-Based Zero-Shot Out-of-Distribution Detection via Self-Learning (Few-shot code)\n")
                print('Usage:')
                print('python few_shot.py -h | --help')
                print("python few_shot.py -M <multiplier> -D <dataset>")
                print('\nOptions:')
                print("-h --help    Show this screen")
                print("-M           Approach multiplier (more details in the paper)")
                print("-D           Dataset name [datasets available below] (Its important to notice that you need to use the simple_features.py code to extract features before start this process)")
                print("\nAvailable datasets (call by the name in the right): ")
                print("CIFAR10:         cifar10")
                print("CIFAR100:        cifar100")
                print("Geological:      geo")
                print("Stanford Cars:   cars")
                print("Caltech-101:     caltech")
                print("Caltech-256:     caltech256")
                print("\nMore details of the approach and the implementation in the paper")
                sys.exit(2)

            elif currentArgument in ("-M"):
                mult = float(currentValue)
            
            elif currentArgument in ("-D"):
                dataset = currentValue
                
        print("Multiplier: ", mult)
        print("Dataset: ", dataset)

    except UnboundLocalError:
        print("You forgot to define something")
        sys.exit(2)
    except getopt.error as err:
        print (str(err))
        sys.exit(2)

    ood_score(mult,dataset)
    


if __name__ == "__main__":
   main()
