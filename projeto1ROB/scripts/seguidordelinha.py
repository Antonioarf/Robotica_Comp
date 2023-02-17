#! /usr/bin/env python
# -*- coding:utf-8 -*-
###########################
#### Variaveis por cor ####
###########################
########LARANJA##########
# ID = 11
# animal = "cat"
# tamanho_minimo = 500
# cor = "Laranja"
#########AZUL##########
# ID = 22
# animal = "dog"
# tamanho_minimo = 300
# cor = AZUL
#########ROSA##########
ID = 13
animal = "bird"
tamanho_minimo = 300
cor = Rosa
__author__ = ["Rachel P. B. Moraes", "Igor Montagner", "Fabio Miranda"]

import os
import rospy
import numpy as np
import tf
import math
import cv2
import time
from geometry_msgs.msg import Twist, Vector3, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage, LaserScan
from cv_bridge import CvBridge, CvBridgeError
import cormodule
import achalinha
import cormodulebranco
from std_msgs.msg import Float64

import cv2.aruco as aruco
import sys
aruco_dict  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters  = aruco.DetectorParameters_create()
parameters.minDistanceToBorder = 0
parameters.adaptiveThreshWinSizeMax = 1000


proto = "/home/borg/catkin_ws/src/Robotica/projeto1ROB/scripts/mobilenet_detection/MobileNetSSD_deploy.prototxt.txt" # descreve a arquitetura da rede
model = "/home/borg/catkin_ws/src/Robotica/projeto1ROB/scripts/mobilenet_detection/MobileNetSSD_deploy.caffemodel" # contém os pesos da rede em si
net = cv2.dnn.readNetFromCaffe(proto, model)
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
CONFIDENCE = 0.7
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
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

bridge = CvBridge()

cv_image = None
media = []
centro = []
atraso = 1.5E9 # 1 segundo e meio. Em nanossegundos

area = 0.0 # Variavel com a area do maior contorno
# Só usar se os relógios ROS da Raspberry e do Linux desktop estiverem sincronizados. 
# Descarta imagens que chegam atrasadas demais
check_delay = False 

# A função a seguir é chamada sempre que chega um novo frame
maior_area_creeper = 0
maior_area = 0
ultimo_aruco = 0
saida = None
resultados = None
def roda_todo_frame(imagem):
	# print("frame")
	global cv_image
	global media
	global centro
	global media_creeper
	global centro_creeper
	global maior_area_creeper
	global maior_area
	global ultimo_aruco
	global saida
	global resultados

	now = rospy.get_rostime()
	imgtime = imagem.header.stamp
	lag = now-imgtime # calcula o lag
	delay = lag.nsecs
	# print("delay ", "{:.3f}".format(delay/1.0E9))
	if delay > atraso and check_delay==True:
		# print("Descartando por causa do delay do frame:", delay)
		return 
	try:
		antes = time.clock()
		cv_image = bridge.compressed_imgmsg_to_cv2(imagem, "bgr8")
		imagem2 = cv_image.copy()
		imagem3 = cv_image.copy()
		# cv_image = cv2.flip(cv_image, -1) # Descomente se for robo real
		media, centro, maior_area =  cormodule.identifica_cor(cv_image)
		media_creeper, centro_creeper, maior_area_creeper = cormodulebranco.identifica_cor(imagem2)
		# print('media')
		# print(maior_area_creeper)
		gray = cv2.cvtColor(imagem3, cv2.COLOR_BGR2GRAY)
		corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
		
		try:
			print(ids[0])
			if ids[0][0] !=100 and ids[0][0] !=200: #and ids[0][0] != 50
				ultimo_aruco = ids[0][0]
		except:
			pass


		# img = cv2.imread(imagem)
		CONFIDENCE = 0.7
		COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
		saida, resultados = detect(imagem3)
		# cv2.imshow("Mobile", saida)
		# print (resultados[0][0])

		depois = time.clock()
		cv2.imshow("GOAL: {},{},{}".format(cor, ID, animal), cv_image)
		cv2.imshow("GOAL: {},{},{}".format(cor, ID, animal), imagem2)
	except CvBridgeError as e:
		print('ex', e)
medida = 0
def scaneou(dado):
	#print("Faixa valida: ", dado.range_min , " - ", dado.range_max )
	#print("Leituras:")
	global medida
	min1 = np.min(dado.ranges[0:3])	
	min2 = np.min(dado.ranges[357:360])
	medida = min([min1,min2])
	# medida = dado.ranges[0]
	# print(medida)
	# print(np.array(dado.ranges).round(decimals=2))
	#print("Intensities")
	#print(np.array(dado.intensities).round(decimals=2))

def achoucreeper(media_var, centro_var,medida, alinhamento,achou):
	# print("aaaaaaaaaaaaaaqui")
	vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
	alinhamento_bool = False
	Vyy = 0
	Vxx = 0.1
	if len(media_var) != 0 and len(centro_var) != 0 and achou == False and medida > 0.17:
		# print("Média dos vermelhos: {0}, {1}".format(media_var[0], media_var[1]))
		# print("Centro_var dos vermelhos: {0}, {1}".format(centro_var[0], centro_var[1]))
		alinhamento = abs(media_var[0] - centro_var[0])
		if media_var[0] != 0:
			if (media_var[0] > centro_var[0]):
				Vyy = -0.2
			if (media_var[0] < centro_var[0]):
				Vyy = 0.2
			if alinhamento<=15:
				alinhamento_bool = True
		else:
			vel = Twist(Vector3(0,0,0), Vector3(0,0,0.1))
	if alinhamento_bool  and  not achou and medida > 0.17:
		Vxx = 0.1
		Vyy = 0
	if  medida <= 0.17: #apaguei a condicao do alinhament
		achou = True
		alinhamento_bool = False
		Vxx = 0
		Vyy = 0
	vel = Twist(Vector3(Vxx,0,0), Vector3(0,0,Vyy))
	velocidade_saida.publish(vel)
	rospy.sleep(0.1)

if __name__=="__main__":
	rospy.init_node("cor")


	# topico_imagem = "/kamera"
	topico_imagem = "/camera/image/compressed" # Use para robo virtual
	# topico_imagem = "/raspicam/image_raw/compressed" # Use para robo real

	recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)
	print("Usando ", topico_imagem)
	recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)
	velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
	braco_publisher = rospy.Publisher('/joint1_position_controller/command', Float64 , queue_size=1)
	garra_publisher = rospy.Publisher('/joint2_position_controller/command',Float64 ,queue_size=1)
	inic = True
	ajeitou = False
	alinhamento = False
	achou = False
	bloco = False
	fez_curva = False
	try:
		pos_braco = Float64()
		pos_braco.data = 1.5
		pos_garra = Float64()
		pos_garra.data = 1

		#Esse bloco sao os publishes que o gazebo ignora
		vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
		velocidade_saida.publish(vel)
		rospy.sleep(0.1)
		braco_publisher.publish(pos_braco)
		garra_publisher.publish(pos_garra)
		rospy.sleep(0.5)

		# as proximas linhas servem apenas como um teste para a garra
	    # para antes do robo comecar o trajeto garantir que estao funcionando
		# nao sendo essenciais para o projeto
		braco_publisher.publish(pos_braco)
		garra_publisher.publish(pos_garra)
		rospy.sleep(0.5)

		pos_braco.data = -1.5
		pos_garra.data = -1

		braco_publisher.publish(pos_braco)
		garra_publisher.publish(pos_garra)
		rospy.sleep(0.5)
		#Esse "if" eh praticamente desnecessario,
		#servindo apenas para agilizar o comeco dos videos
		if ID == 22:
			vel = Twist(Vector3(0,0,0), Vector3(0,0,0.3))
			velocidade_saida.publish(vel)
			rospy.sleep((math.pi/0.6))
		elif ID == 11:
			vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.3))
			velocidade_saida.publish(vel)
			rospy.sleep(math.pi/0.6)

		vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
		Vx = 0
		Vy = 0
		delay = 0.1
		while not rospy.is_shutdown():
			try:
				for e in resultados:
					print (e[0])
					if e[0] == animal and ajeitou:
						bloco = True
						output_certo = e
						
			except TypeError:
				pass
			
			if ajeitou and bloco:
				
				try:
					meio_x = (output_certo[2][0] + output_certo[3][0])/2
					meio_y = (output_certo[2][1] + output_certo[3][1])/2
	
					if (meio_x - centro[0]>15):
						# print('direita')
						Vy = -0.1
						
					elif (meio_x - centro[0]<15):
						# print('esquerda')
						Vy = 0.1
				
					else:
						alinhamento = True	
						Vx = 0.1
						Vy = 0
					if alinhamento:
						Vx = 0.1
						
				except:
					pass
				if medida <= 0.50:
					vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
					velocidade_saida.publish(vel)
					rospy.sleep(0.1)

					pos_braco.data = 0
					braco_publisher.publish(pos_braco)
					rospy.sleep(0.5)
							
					pos_garra.data = -1
					garra_publisher.publish(pos_garra)
					rospy.sleep(0.5)

					pos_braco.data = -1.5
					braco_publisher.publish(pos_braco)
					rospy.sleep(0.5)

					vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.2))
					velocidade_saida.publish(vel)
					rospy.sleep(5)
					print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
					ajeitou = False
			elif maior_area <= 20 and maior_area_creeper < tamanho_minimo/3:
				Vx = 0
				Vy = -0.3
				if ultimo_aruco == 150 or ultimo_aruco == 50:
					delay = 8.5
				else:
					delay = math.pi/0.3 
 			elif maior_area_creeper > tamanho_minimo and ultimo_aruco == ID and not ajeitou:
				achou = True
				achoucreeper(media_creeper, centro_creeper,medida, alinhamento,False)
			elif len(media) != 0 and len(centro) != 0 : #Vy = 0
				alinhamento = abs(media[0] - centro[0])
				if media[0] != 0:
					if (media[0] - centro[0]>15):
						# print('direita')
						Vy = -0.1
						
					elif (media[0] - centro[0]<15):
						# print('esquerda')
						Vy = 0.1
				
					else:
						alinhamento = True	
						Vx = 0.1
						Vy = 0

					if alinhamento:
						Vx = 0.1
			elif not achou:
				# print('nao achou')
				Vx = 0
				Vy = 0.1

			vel = Twist(Vector3(Vx,0,0), Vector3(0,0,Vy))
			velocidade_saida.publish(vel)
			rospy.sleep(delay)
			Vx = 0
			Vy = 0
			delay = 0.1
			if medida <= 0.30 and not ajeitou:
				pos_braco.data = -0.3
				braco_publisher.publish(pos_braco)
				rospy.sleep(0.5)
				if medida <= 0.17 : 
					print("#########################################")
					vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
					velocidade_saida.publish(vel)
					rospy.sleep(0.1)

					pos_garra.data = 1
					garra_publisher.publish(pos_garra)
					rospy.sleep(0.5)

					pos_braco.data = 1.5
					braco_publisher.publish(pos_braco)
					rospy.sleep(0.5)

					Vx = 0
					Vy = -0.3
					delay = 3
					vel = Twist(Vector3(Vx,0,0), Vector3(0,0,Vy))
					velocidade_saida.publish(vel)
					rospy.sleep(delay)
					achou = False
					ajeitou = True
			if not fez_curva and ajeitou and ID == 11 and ultimo_aruco == 22:
					# print ("DEVERIA ESTAR INDO RETO")
					Vxx = 0.3
					Vyy = 0
					delayy = 1
					vell = Twist(Vector3(Vxx,0,0), Vector3(0,0,Vyy))
					velocidade_saida.publish(vell)
					rospy.sleep(delayy)
					# print ("DEVERIA ESTAR VIRANDO")
					Vxx = 0
					Vyy = -0.3
					delayy = 1.5
					vell = Twist(Vector3(Vxx,0,0), Vector3(0,0,Vyy))
					velocidade_saida.publish(vell)
					rospy.sleep(delayy)
					fez_curva = True
					# print ("DEVERIA ESTAR PARANDO DE GIRAR")
	except rospy.ROSInterruptException:
	    print("Ocorreu uma exceção com o rospy")


		# braco_publisher.publish(pos_braco) 0 a pi
		# garra_publisher.publish(pos_garra) 0 a 1
		# braco_publisher = rospy.Publisher('/joint1_position_controller/command', Float64 , queue_size=1)
		# garra_publisher = rospy.Publisher('/joint2_position_controller/command',Float64 ,queue_size=1)