import cv2 as cv
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split

sift = cv.SIFT_create()

if not(Path('model.joblib').exists()):
    des = []
    for file in Path('train_images').glob('*'):
        _,curDes = sift.detectAndCompute(cv.cvtColor(cv.imread(str(file)),cv.COLOR_BGR2GRAY),None)
        for i in curDes:
            des.append(i)
    des = np.array(des)
    des_model = KMeans(n_clusters=1024).fit(train_test_split(des,test_size=0.2)[1])
    dump(des_model, 'model.joblib')

model = load('model.joblib')

def vectorize(img):
    _,ides = sift.detectAndCompute(cv.cvtColor(img,cv.COLOR_BGR2GRAY),None)
    classes = model.predict(ides)
    hist,_ = np.histogram(classes,1024,[0,1024])
    return 1024 * hist / sum(hist)


if not(Path('base.csv').exists()):
    paths = []
    vecs = []
    for file in Path('base').glob('*'):
        paths.append(file)
        vecs.append(vectorize(cv.imread(str(file))).tolist())
    df = pd.DataFrame(data={'Path':paths, 'Vec':vecs})
    df = df.reset_index()
    df = df.rename(columns={'index':'Id'})
    df.to_csv('base.csv',index=False)
