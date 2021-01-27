# -*- coding: utf-8 -*-

from skimage import metrics 
import cv2 
import argparse
import os
import glob

argument = argparse.ArgumentParser()  
argument.add_argument("-i", "--input", type=str, required=True,
	help="sciezka do katalogu z obrazami wejsciowymi")
argument.add_argument("-e", "--extension", type=str, 
    required=True, help="rozszerzenie plików obrazowych")
arguments = vars(argument.parse_args())
input_images = []
images_names = []
grayscale_images = []

os.chdir(arguments["input"]) 
for file in glob.glob("*." + arguments["extension"]):
    image = cv2.imread(file)
    input_images.append(image)
    images_names.append(file)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_images.append(grayscale_image)

original = input_images[0]
original_gray = grayscale_images[0]

string1 = "SSIM dla obrazu "
string2 = " = "

k=0
for gray in grayscale_images:
    dim = (original_gray.shape[1],original_gray.shape[0])
    gray = cv2.resize(gray, dim, cv2.INTER_NEAREST)
    if(k != 0):
        (score, diff) = \
        metrics.structural_similarity(original_gray, 
                                      gray, full=True)
        diff = (diff * 255).astype("uint8")
        score = round(score, 2)
        print(string1 + images_names[k] + string2 + str(score))
    k+=1
    



 

