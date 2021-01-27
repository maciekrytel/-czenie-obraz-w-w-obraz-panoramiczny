# -*- coding: utf-8 -*-

from math import log10, sqrt 
import cv2 
import numpy
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

os.chdir(arguments["input"]) 
for file in glob.glob("*." + arguments["extension"]):
    image = cv2.imread(file)
    input_images.append(image)
    images_names.append(file)

original = input_images[0]
max_pixel = 255.0
k=0
string1 = "PSNR dla obrazu "
string2 = " = "
string3 = " [dB]"

for panoramas in input_images:
    name = images_names[k]
    dim = (original.shape[1],original.shape[0])
    panoramas = cv2.resize(panoramas, dim, cv2.INTER_NEAREST)
    mse = numpy.mean((original - panoramas) ** 2) 
    if (mse != 0):
        psnr = 20*log10(max_pixel / sqrt(mse))
        psnr = round(psnr, 2)
        print(string1 + images_names[k] + string2
              + str(psnr) + string3)
    k+=1


 

