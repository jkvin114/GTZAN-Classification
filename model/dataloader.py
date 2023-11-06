
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import torch

genres={'pop':1,'classical':2,'reggae':3,'disco':4,'jazz':5,'metal':6,'country':7,'blues':8,'hiphop':9,'rock':0}


def load_data(limit=None):
    
    MUSIC = './dataset/Data/images_original'
    music_paths = []
    genre_target = []
    for root, dirs, files in os.walk(MUSIC):
        for name in files:
            filename = os.path.join(root, name)
            
            if filename != '/dataset/Data/images_original/jazz/jazz00054.png':
                music_paths.append(filename)
                genre_target.append(root.split('\\')[1])
                

    music_arr=[]
    genre_new=[]
    count = len(music_paths)
    print(count) #999


    if limit is not None:
        music_paths=music_paths[:limit]

    # Define the crop coordinates (left, upper, right, lower)
    crop_box = (55, 36, 390, 252)  # Adjust these values to define your desired crop

    for idx, spec_path in enumerate(music_paths):
        image = Image.open(spec_path)  

        # Crop the image
        cropped_image = image.crop(crop_box)

        resized_image = cropped_image.resize((128,128), Image.LANCZOS)
        #resized_image.show()
        image_arr=np.array(resized_image)[:,:,:3]
        music_arr.append(np.moveaxis(image_arr,2,0))

        genre_new.append(genre_target[idx])


    genre_id = [genres[item] for item in genre_new]
    
    X_train, X_test, y_train, y_test = train_test_split(music_arr, genre_id, test_size=0.15, random_state=1)

    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

    # create feature and targets tensor
    torch_X_train = torch.FloatTensor(X_train)
    torch_y_train = torch.LongTensor(y_train)

    
    torch_X_test = torch.FloatTensor(X_test)
    torch_y_test = torch.LongTensor(y_test)
    
    torch_X_val = torch.FloatTensor(X_val)
    torch_y_val = torch.LongTensor(y_val)


   # torch_X_train=F.normalize(torch_X_train,dim=0)
   # torch_X_test=F.normalize(torch_X_test,dim=0)
   # torch_X_val=F.normalize(torch_X_val,dim=0)

    print("Train shape:")
    print(torch_X_train.size())

    print("Val shape:")
    print(torch_X_val.size())
    print("Test shape:")
    print(torch_X_test.size())

    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
    test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)
    val = torch.utils.data.TensorDataset(torch_X_val,torch_y_val)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = 128, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test, batch_size = 32, shuffle = False)
    val_loader = torch.utils.data.DataLoader(val, batch_size = 32, shuffle = False)
    


    loaders={
        "train":train_loader,
        "test":test_loader,
        "val":val_loader
        }

    return loaders