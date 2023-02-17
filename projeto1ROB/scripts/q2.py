#!/usr/bin/python
# -*- coding: utf-8 -*-

# Este NÃO é um programa ROS

from __future__ import print_function, division 

import cv2
import os,sys, os.path
import numpy as np

print("Rodando Python versão ", sys.version)
print("OpenCV versão: ", cv2.__version__)
print("Diretório de trabalho: ", os.getcwd())

# Arquivos necessários
# Baixe e salve na mesma pasta que este arquivo
# https://github.com/Insper/robot20/raw/master/media/animais_caixas.mp4
video = "animais_caixas.mp4"

## CAIXA VERDE
## CAO 
proto = "./mobilenet_detection/MobileNetSSD_deploy.prototxt.txt" # descreve a arquitetura da rede
model = "./mobilenet_detection/MobileNetSSD_deploy.caffemodel" # contém os pesos da rede em si
net = cv2.dnn.readNetFromCaffe(proto, model)
CONFIDENCE = 0.7
CLASSES = ["dog","background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	 "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def pontos(contorno):
    maior = None
    maior_area = 0
    for c in contornos:
        area = cv2.contourArea(c)
        if area > maior_area:
            maior_area = area
            maior = c
    maiorX = 0
    maiorY = 0
    menorX = 0
    menorY = 0
    for e in maior:
        if e[0][1] > maiorX:
            maiorX = e[0][1]
        if e[0][0] > maiorY:   
            maiorY = e[0][0]
        if e[0][1] > menorX:
            menorX = e[0][1]
        if e[0][0] > menorY:
            menorY = e[0][0]
    return [maiorX,menorX,maiorY,menorY]

def detect(frame):
    """
        Recebe - uma imagem colorida BGR
        Devolve: objeto encontrado
    """
    image = frame.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    # print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    results = []

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence


        if confidence > CONFIDENCE:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            # print("[INFO] {}".format(label))
            cv2.rectangle(image, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            results.append((CLASSES[idx], confidence*100, (startX, startY),(endX, endY) ))

    # show the output image
    return image, results


if __name__ == "__main__":

    # Inicializa a aquisição da webcam
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_FPS, 3)


    print("Se a janela com a imagem não aparecer em primeiro plano dê Alt-Tab")

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if ret == False:
            #print("Codigo de retorno FALSO - problema para capturar o frame")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            #sys.exit(0)
        
        # Our operations on the frame come here
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        saida, resultados = detect(rgb)
        saida_rgb = cv2.cvtColor(saida, cv2.COLOR_BGR2RGB)
        b_epi, g_epi, r_epi = cv2.split(rgb)
        
        cor_menor = np.array([30, 50, 50])
        cor_maior = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, cor_menor, cor_maior)
        segmentado = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((4, 4)))
        contornos, arvore = cv2.findContours(segmentado.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        
        pp = pontos(contornos)
        if len(resultados) !=0 :
           if (resultados[0][2][0] <= pp[0]) and  (resultados[0][2][1] <= pp[2]) and (resultados[0][3][0] <= pp[1]) and (resultados[0][3][1] <= pp[3]):
                print("DENTRO")

        # NOTE que em testes a OpenCV 4.0 requereu frames em BGR para o cv2.imshow
        cv2.imshow('imagem', frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


