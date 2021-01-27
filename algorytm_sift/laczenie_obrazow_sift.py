# -*- coding: utf-8 -*-

import cv2 #biblioteka OpenCV
import numpy #biblioteka NumPy
import argparse #moduł dla commandlinow
import os #umozliwia operowania na plikach i folderach
import glob #biblioteka glob


argument = argparse.ArgumentParser() #tworzymy argument 
argument.add_argument("-i", "--input", type=str, required=True,
	help="sciezka do katalogu z obrazami wejsciowymi") #dodawany jest argument, ktory przechowuje sciezke folder z plikami do polczenia
argument.add_argument("-e", "--extension", type=str, required=True,
	help="rozszerzenie plików obrazowych") #dodawany jest argument z rozszerzeniami plikow czekajacych na polaczenie
argument.add_argument("-o", "--from_right_to_left?", type=str, required=True,
	help="z której strony doklejane mają być zdjęcia [yes or no]") #dodawany jest argument z kolejnoscia laczenia obrazow
arguments = vars(argument.parse_args())

#tworzone są puste listy na obrazy wejsciowe 
#i ich monochromatyczne odpowiedniki
input_images = []
grayscale_images = []

os.chdir(arguments["input"]) #zmieniamy obecny katalog na ten zadany przez uzytkownika
for file in glob.glob("*." + arguments["extension"]): #w petle ladujemy kazdy z plikow znajdujacy sie w katalogu i dodajemy go do listy obrazow wejsciowych
    image = cv2.imread(file)
    input_images.append(image)
    
    #kazdy z obrazow wejsciowych konwertowany jest na skale szarosci
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grayscale_images.append(grayscale_image)

#w zalenosci czy obrazy wejsciowe sa ulozone w kolejnosci
#od lewej do prawej, badz tez na odwrot, lista plikow
#zostanie odpowiednio posortowana
if(arguments["from_right_to_left?"] == "yes"):
    input_images = input_images[::-1]
    grayscale_images = grayscale_images[::-1]    

#tworzone sa teraz puste listy, w ktorych beda
#przechowywane punkty kluczowe i deskryptory
keypoints = []
descriptors = []

#teraz zostana okreslone punkty kluczowe i deskryptory
#dla kazdego z obrazow wejsciowych przy wykorzystaniu
#algorytmu do detekcji cech obrazu opisanego w algorytmie SIFT
sift = cv2.xfeatures2d.SIFT_create()
print("[INFO] Obliczanie punktow kluczowych" + 
      "i deksryptorow obrazow... \n")
for grayscale_image in grayscale_images:
    kp, des = sift.detectAndCompute(grayscale_image,None)
    keypoints.append(kp)
    descriptors.append(des)

#teraz nalezy przejsc do dopasowywania cech
#pierwszy obraz na liscie plikow wejsciowych
#oznaczony indeksem nr_of_img zostanie porownany
# z kolejnym obrazem z listy z indeksem nr_of_train_img
#jezeli dane obrazy nie beda do siebie pasowaly
#obraz pierwszy bedzie porownywany z kolejnymi
#az do znalezienia wspolnych cech
nr_of_img=0
nr_of_train_img = 1
while nr_of_img < len(input_images) \
and nr_of_train_img < len(input_images):
    
    info = "[INFO] laczenie obrazu nr "
    i_str = " i nr " 
    print(info+str(nr_of_img) + i_str + str(nr_of_train_img)) 
    
    #dopasowywanie cech dwoch obrazow
    #odbywa sie za pomoca Brut-Force Matchera
    #czyli dla kazdego deksryptora z pierwszego
    #obrazu znajdowany jest najblizszy deskryptor
    #w drugim obrazie
    matches = []
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors[nr_of_img],
                          descriptors[nr_of_train_img], k=2)
    
    #w celu wyeliminowania trywialnych dekryptorow
    #przeprowadzany jest prosty test eleimujacy
    #takie polaczenia    
    good = []
    for m in matches:
        if (m[0].distance < 0.5*m[1].distance):
            good.append(m)
    
    #po znalezeniu dopasowan miedzy obrazami
    #wykonywane jest sprawdzenie ile takowych jest
    #jezeli jest ich wiecej niz 4,
    #mozna dokonac polaczenia
    matches = numpy.asarray(good)
    src = []
    if (len(matches[:,0]) >= 4):
        
        #Aby polczyc oba obrazy nalezy wyznaczyc macierz homografii
        src = \
        numpy.float32([ keypoints[nr_of_img][m.queryIdx].pt \
                             for m in matches[:,0] \
                             ]).reshape(-1,1,2)
        dst = \
        numpy.float32([ keypoints[nr_of_train_img][m.trainIdx].pt\
                             for m in matches[:,0] \
                             ]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        
        #Majac juz macierz homografii, mozliwe jest 
        #znieksztalcenie oraz zszycie obrazu
        dst = \
        cv2.warpPerspective(input_images[nr_of_img],H,
                    ((input_images[nr_of_img].shape[1] + 
                    input_images[nr_of_train_img].shape[1]), 
                    input_images[nr_of_train_img].shape[0]))
        dst[0:input_images[nr_of_train_img].shape[0], 
            0:input_images[nr_of_train_img].shape[1]] = \
            input_images[nr_of_train_img]
        
        #polaczone dwa obrazy beda zszywane z kolejnymi
        #obrazami wejsciowymi. W tym celu nalezy wyznaczyc
        #punkty kluczowe i deksryptory zszytego obrazu
        grayscale_image = \
        cv2.cvtColor(input_images[nr_of_train_img],
                     cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(grayscale_image,None)
        keypoints[nr_of_train_img] = kp
        descriptors[nr_of_train_img] = des
        
        #w celu pozbawienia laczonych obrazow czarnych obszarow
        #najpierw zostanie wykonane odwrocone progowanie binarne
        #w celu uwtorzenia prostokata zaweirajacego
        #wszystkie czarne piksele
        grayscale_image = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        
        ret,thresh_black = \
        cv2.threshold(grayscale_image,0,255,cv2.THRESH_BINARY_INV)
        
        kernel_bl = numpy.ones((5,5), numpy.uint8)
        thresh_black = \
        cv2.morphologyEx(thresh_black, cv2.MORPH_CLOSE, kernel_bl)
        thresh_black = \
        cv2.morphologyEx(thresh_black, cv2.MORPH_OPEN, kernel_bl)
        
        contours_bl = \
        cv2.findContours(thresh_black, 
                         cv2.RETR_EXTERNAL, 
                         cv2.CHAIN_APPROX_SIMPLE)
        contours_bl = \
        contours_bl[0] \
        if len(contours_bl) == 2 else contours_bl[1]
        for cntr in contours_bl:
            a,b,c,d = cv2.boundingRect(cntr)

        #majac kontury prostokata zawierajacego czarne pikele
        #chcemu wydobyc kontury prostokata zawierajacego
        #wszytskie niezerowe piksele.
        _,thresh = \
        cv2.threshold(grayscale_image,0,255,cv2.THRESH_BINARY)
        
        kernel = numpy.ones((5,5), numpy.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours =\
        cv2.findContours(thresh,
                         cv2.RETR_EXTERNAL, 
                         cv2.CHAIN_APPROX_SIMPLE)
        contours = \
        contours[0] if len(contours) == 2 else contours[1]
        for cntr in contours:
            x,y,w,h = cv2.boundingRect(cntr)
            
            #teraz dokonywane jest dociecie panoramy
            #szerokosc panoramy wyznacza jest od
            #poczatku niezerowego prostokata
            #do poczatku zerowego prostoka
            #w ten sposob sa eleiminowane wszystkie
            #czarne obszary
            crop = dst[y:y+h,x:a]

        input_images[nr_of_train_img] = crop
        nr_of_img+=1
        nr_of_train_img= nr_of_img + 1
    else:
        nr_of_train_img+=1

#jezeli dokonane zostalo choc jedno zszycie
#mozliwe jest utworzenie panoramy
if('crop' in locals()):
    
    #panorama zostaje zapisana i wyswietlona
    #na koniec pracy programu
    cv2.imwrite('panorama_sift.jpg',crop)
    print("[INFO] panorama gotowa")
    image = cv2.imread('output.jpg')
    cv2.imshow("panorama", crop)
    cv2.waitKey(0)
else:
    raise AssertionError('[INFO] nie mozna zszyc')
        
