#!/usr/bin/python
# -*- coding: utf-8 -*-

import cv2
import sys
import auxiliar as aux
import numpy as np
import math 
def m(y1,y2,x1,x2):
    m= (y2-y1) / (x2-x1)
    return m

def calcula_h(ay, m, ax):
    h = ay - m * ax
    return h
    
def calcula_ponto(h1,h2,m1,m2):
    x = (h1-h2)/(m2-m1)
    y = m1*x+h1
    return(x,y)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        try:
            input_source=int(arg) # se for um device
        except:
            input_source=str(arg) # se for nome de arquivo
    else:   
        input_source = 0

    cap = cv2.VideoCapture('line_following.mp4')

    esquerda = 0
    x_esq = 0
    y_esq= 0
    direita = 0
    x_dir = 0
    y_dir = 0
    
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == False:
            print("Codigo de retorno FALSO - problema para capturar o frame")

        # Our operations on the frame come here
        imagem = frame.copy()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)


        cor_menor = np.array([0, 0, 200 ])
        cor_maior = np.array([180, 10, 255])
        mask_coke = cv2.inRange(img_hsv, cor_menor, cor_maior)

        edges = cv2.Canny(mask_coke, 50, 200)
        # Detect points that form a line
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=5, maxLineGap=150)
        # Draw lines on the image
        coeficientes = []
        lista_x = []
        lista_y=[]
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            m= (y2-y1) / (x2-x1)
            if abs(m) <1000:
                if m>0:
                    cv2.line(imagem, (x1, y1), (x2, y2), (90, 0, 0), 9)
                else:
                    cv2.line(imagem, (x1, y1), (x2, y2), (0, 90, 0), 9)
                coeficientes.append(m)
                lista_x.append(x1)
                lista_y.append(y1)
        
        

        contador = 0
        for coef in coeficientes:
            # print (coef)
            if coef > 0:
                direita = coef
                x_dir = lista_x[contador]
                y_dir = lista_y[contador]
            elif coef <0:
                esquerda = coef
                x_esq = lista_x[contador]
                y_esq = lista_y[contador]
            contador+=1

        
        h_esq = calcula_h(y_esq,esquerda,x_esq)
        h_dir =  calcula_h(y_dir,direita,x_dir)
        if h_dir != h_esq and esquerda != direita and esquerda != 0 and direita != 0 :
            Xponto,Yponto = calcula_ponto(h_esq,h_dir,esquerda,direita)
            cv2.circle(imagem, (int(Xponto),int(Yponto)), 10, (0,0,255), 10)
        # print("o x é {0} e o y é {1}".format(Xponto,Yponto))




        # ret,limiarizada = cv2.threshold(gray,200,200,cv2.THRESH_OTSU)

        # Display the resulting frame
        # cv2.imshow('frame',frame)
        cv2.imshow('frame', imagem)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

