import pandas as pd
import os
import SimpleITK as sitk
import numpy as np
import cv2
from sklearn.model_selection import train_test_split


def generate(fold):
    dire = []
    label = []
    for filename in os.listdir(fold):
        if "cat" in filename:
            dire.append(fold + filename)
            label.append(0)
        elif "dog" in filename:
            dire.append(fold + filename)
            label.append(1)

    X_train, X_test, y_train, y_test = train_test_split(dire, label, test_size=0.2)
    d = {'dire': X_train, 'label': y_train}
    training = pd.DataFrame(data=d)
    d1 = {'dire': X_test, 'label': y_test}
    test = pd.DataFrame(data=d1)
    training.to_csv("C:/Users/alexn/Desktop/PetImages/Train.csv", index=False)
    test.to_csv("C:/Users/alexn/Desktop/PetImages/Test.csv", index=False)


if __name__ == '__main__':
    folder = "C:/Users/alexn/Desktop/PetImages/train/"
    generate(folder)
