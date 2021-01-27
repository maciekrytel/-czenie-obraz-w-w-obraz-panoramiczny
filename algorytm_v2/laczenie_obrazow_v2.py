"""
Created on Tue Dec 15 16:11:35 2020

@author: Komputer
"""

# -*- coding: utf-8 -*-

import cv2 #biblioteka OpenCV
import numpy #biblioteka NumPy
import imutils #biblioteka Imutils
import glob #biblioteka glob
import argparse #moduł dla commandlinów
import os

argument = argparse.ArgumentParser() #tworzymy argument 
argument.add_argument("-i", "--input", type=str, required=True,
	help="sciezka do katalogu z obrazami wejsciowymi") #dodawany jest argument, który przechowuje sciezke folder z plikami do polczenia
argument.add_argument("-e", "--extension", type=str, 
    required=True, help="rozszerzenie plików obrazowych") #dodawany jest argument z rozszerzeniami plikow czekajacych na połączenie
arguments = vars(argument.parse_args())
input_images = []

os.chdir(arguments["input"]) #zmieniamy obecny katalog na ten zadany przez użytkownika
for file in glob.glob("*." + arguments["extension"]): #w pętle ładujemy kązdy z plików znajdujący się z katalogu i dodajemy go do listy obrazów wejsciowych
    image = cv2.imread(file)
    input_images.append(image)
    
#tutaj tworzony jest obiekt "zszywania" i realizowane jest zszywanie za pomoca biblioteki OpenCV
stitcher = cv2.createStitcher() if imutils.is_cv3() \
else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(input_images)


#status 0 oznacza poprawne wykonanie zszycia
if status == 0:
    #tworzymy obramówke dla stworzonej panoramy o szerokosci 10 pixeli
    print("[INFO] przycinanie...")
    stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
        cv2.BORDER_CONSTANT, (0, 0, 0))

    #utworzona panorama zostaje przekonwertowana na skale szarosci 'gray'
    #a następnie mamy progowanie obrazu 'thresh', 
    #gdzie kazdemu pixelowi wiekszemu niz zero przypisujemy wartosc 255
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    
    #teraz znajdujemy kontury naszej panoramy
    #wybieramy obrys o największej powierzchni
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    #posiadając już kontury, możemy utworzyć ramkę do której będzie dopasowana
    #nasza panorama
    mask = numpy.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    
    #tutaj tworzymy dwie kopie naszej maski
    #pierwsza będzie zmniejszana w pętli while
    #druga bedzie wykorzystana do sprawdzenia czy wewnatrz ciagle zmniejszanej
    #maski zostaną pixele które są równe 0, jezeli ich nie będzie, kończymy
    #zmniejszanie minRect
    minRect = mask.copy()
    sub = mask.copy()
    
    #pętla while bedzie sie wykonywala az do momentu, gdy w maske sub
    #nie zostanie zaden pixel rowny 0
    while cv2.countNonZero(sub) > 0:

        minRect = cv2.erode(minRect, None) #zmiejszamy maskę
        sub = cv2.subtract(minRect, thresh)#nakłądamy zmiejszona maske minRect na maske Sub
        
    #znajdujemy kontury w minimalnej masce prostokatnej
    #i podajemy wspołrzędcne ramki w którą dopasujemy panoramę 
    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    
    #zapisujemy i wyswietlamy uzyskane panoramy
    stitched_v2 = stitched[y:y + h, x:x + w]
    cv2.imwrite("panorama_v2.jpg", stitched_v2)
    cv2.imshow("panorama_v1", stitched_v2)
    cv2.waitKey(0)
    
#gdy status jest inny niż 0, oznacza to że wystąpił błąd
else:
    print("łączenie nie powiodło się")
    

