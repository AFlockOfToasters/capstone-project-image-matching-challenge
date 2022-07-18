import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.svm import LinearSVR

def load_scenes(scenes, datadir):
    
    """
    load the image data from the correcsponding directory, together with the pairing metrics.
    Args:
        scenes:     List of scenes (folder names under /train)
        datadir:    String of the directory to source image data from

    Returns:
        df: dataframe containing the image pairings and other metrics
        images: loaded images dataframe
    """    
    
    df = pd.DataFrame()
    images = pd.DataFrame()
    # path which contains all the categories of images
    j = 0
    for i in scenes:                                    # durch alle o.g. kategorien hinweg
        print(f'loading category {j+1} of {len(scenes)}: {i}')
        j += 1
        targetpath = os.path.join(datadir,i,"pair_covisibility.csv")
        if df.empty:
            df = pd.read_csv(targetpath)
            df["building"] = f"{i}"
        else:
            dfappend = pd.read_csv(targetpath)
            dfappend["building"] = f"{i}"
            df = pd.concat([df,dfappend],axis=0)
        imgpath=os.path.join(datadir,i,'images')           # Dateipfad der Bilder
        flat_data_arr=[]
        for img in os.listdir(imgpath):                    # für jedes Bild im jeweiligen dateipfad:
            img_array=imread(os.path.join(imgpath,img))    # bild als Array einlesen
            img_resized=resize(img_array,(150,150,3))   # bild resizen, um Einheitlichkeit zu haben
            flat_data_arr.append(img_resized.flatten()) # array zum Vektor glätten   
        print(f'loaded {i} successfully')


        flat_data=np.array(flat_data_arr)
        if images.empty:
            images=pd.DataFrame(flat_data) #dataframe
            #images["imgarray"] = flat_data
            images["image_id"] = [image_id.split(".")[0] for image_id in os.listdir(imgpath)]
            images["building"] = f"{i}"
        else:
            imagesappend = pd.DataFrame(flat_data)
            #imagesappend["imgarray"] = flat_data
            imagesappend["building"] = f"{i}"
            imagesappend["image_id"] = [image_id.split(".")[0] for image_id in os.listdir(imgpath)]
            images = pd.concat([images, imagesappend], axis=0)
            
    df[["image1_id", "image2_id"]] = [pair.split("-") for pair in df.pair]
    df[["fm1","fm2","fm3","fm4","fm5","fm6","fm7","fm8","fm9"]] = pd.DataFrame([fm.split() for fm in df.fundamental_matrix]).astype(float)
    return df, images

def inflate_with_images(df, images):
    """
    loads the corresponding image vectors from a dataframe of flattened images to each image of an image pair and merges them into the input df.

    Args:
        df: dataframe containing the image pairings and other metrics
        images: images dataframe

    Returns:
        df (pd.DataFrame): Inflated df dataframe
    """    
    print("inflating 🐡")
    # merge image 1
    df = pd.merge(df,images.drop(["building"],axis=1),how="left", left_on="image1_id",right_on="image_id",suffixes=["","_1"]).drop("image_id",axis=1)
    #merge image 2
    df = pd.merge(df,images.drop(["building"],axis=1),how="left", left_on="image2_id",right_on="image_id",suffixes=["","_2"]).drop("image_id",axis=1).drop(["image1_id", "image2_id"],axis=1)

    print("done inflating 💥")
    return df

def fit_pred_9xLinSVR(x_train,x_test,y_train, images):
    
    """
    Inflates (loads the corresponding image vectors to each image pair) and then fits the training data to 9 different Linear support vector regressor (LinearSVR) models. 1 for each value of the fundamental matrix
    
    Args:
        x_train:    Training feature dataframe, containing the columns "image1_id", "image2_id", "pair", "building".
        x_test:     Test feature dataframe, containing the columns "image1_id", "image2_id", "pair", "building".
        y_train:    Training targets, containing the columns "fm1" to "fm9".
        images:     Image dataframe, containing flattened image data and an "image_id" column, necessary to merge images onto the training datasets.

    Returns:
        y_pred: dataframe containing all entries of the predicted fundamental matrices as separate columns "fm1" to "fm9".
    """

    models = []
    print(f"fitting dataset of {x_train.shape[0]} training entries")
    print(f"then predicting {x_test.shape[0]} test entries")
    print("-"*30)
    print(f"inflating {x_train.shape[0]+x_test.shape[0]} entries with images")
    print("training data:")
    x_train_inflated = inflate_with_images(x_train, images)
    print("test data:")
    x_test_inflated = inflate_with_images(x_test, images)
    print("-"*30)
    print("initialising models... 🤖")
    for i in range(1,10):
        exec(f"model{i} = LinearSVR(random_state=0, C=0.1)")    # technically bad practice, but so is using 9 SVMs for a NN-Job :-P 
        exec(f"models.append(model{i})")                        
    print("-"*30)
    y_pred = pd.DataFrame()
    for i, model in enumerate(models):
        print(f"fitting model{i+1} for fm{i+1} 📐")
        model.fit(x_train_inflated.iloc[:,2:],y_train[f"fm{i+1}"])
        print("predicting 🔮")
        temppred = model.predict(x_test_inflated.iloc[:,2:])
        y_pred[f"fm{i+1}_pred"] = np.array(temppred)
        print("-"*30)
    return y_pred