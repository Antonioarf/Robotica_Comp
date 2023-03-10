#! /usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = ["Rachel P. B. Moraes", "Igor Montagner", "Fabio Miranda"]


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
import cormodulebranco
import auxiliar as aux


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
def roda_todo_frame(imagem):
	# print("frame")
	global cv_image
	global media
	global centro

	now = rospy.get_rostime()
	imgtime = imagem.header.stamp
	lag = now-imgtime # calcula o lag
	delay = lag.nsecs
	# print("delay ", "{:.3f}".format(delay/1.0E9))
	if delay > atraso and check_delay==True:
		print("Descartando por causa do delay do frame:", delay)
		return 
	try:
		antes = time.clock()
		cv_image = bridge.compressed_imgmsg_to_cv2(imagem, "bgr8")
		# cv_image = cv2.flip(cv_image, -1) # Descomente se for robo real
		media, centro, maior_area =  cormodulebranco.identifica_cor(cv_image)
		print(media)
		depois = time.clock()
		cv2.imshow("Camera", cv_image)
	except CvBridgeError as e:
		print('ex', e)

medida = -1
def scaneou(dado):
	#print("Faixa valida: ", dado.range_min , " - ", dado.range_max )
	#print("Leituras:")
	global medida
	min1 = np.min(dado.ranges[0:3])	
	min2 = np.min(dado.ranges[357:360])
	medida = min([min1,min2])
	# print(medida)
	# print(np.array(dado.ranges).round(decimals=2))
	#print("Intensities")
	#print(np.array(dado.intensities).round(decimals=2))
def achoucreeper(media, centro,medida, alinhamento,achou):
	vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
	if len(media) != 0 and len(centro) != 0 and achou == False:
		# print("Média dos vermelhos: {0}, {1}".format(media[0], media[1]))
		# print("Centro dos vermelhos: {0}, {1}".format(centro[0], centro[1]))
		alinhamento = abs(media[0] - centro[0])
		if media[0] != 0:
			if (media[0] > centro[0]):
				vel = Twist(Vector3(0,0,0), Vector3(0,0,-0.02))
			if (media[0] < centro[0]):
				vel = Twist(Vector3(0,0,0), Vector3(0,0,0.02))
				
			if alinhamento<=8:
				alinhamento = True
		else:
			vel = Twist(Vector3(0,0,0), Vector3(0,0,0.1))
			
	velocidade_saida.publish(vel)
	rospy.sleep(0.1)
	if alinhamento == True and achou == False:
		vel = Twist(Vector3(0.8,0,0), Vector3(0,0,0))
		velocidade_saida.publish(vel)
		rospy.sleep(0.1)

	if alinhamento == True and medida <= 0.5:
		achou = True
		velocidade = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
		velocidade_saida.publish(velocidade)
		rospy.sleep(10)
if __name__=="__main__":
	rospy.init_node("cor")

	# topico_imagem = "/kamera"
	topico_imagem = "/camera/rgb/image_raw/compressed" # Use para robo virtual
	# topico_imagem = "/raspicam/image_raw/compressed" # Use para robo real
	
	# Para renomear a *webcam*
	#   Primeiro instale o suporte https://github.com/Insper/robot19/blob/master/guides/debugar_sem_robo_opencv_melodic.md
	#
	#	Depois faça:
	#	
	#	rosrun cv_camera cv_camera_node
	#
	# 	rosrun topic_tools relay  /cv_camera/image_raw/compressed /kamera
	#
	# 
	# Para renomear a câmera simulada do Gazebo
	# 
	# 	rosrun topic_tools relay  /camera/rgb/image_raw/compressed /kamera
	# 
	# Para renomear a câmera da Raspberry
	# 
	# 	rosrun topic_tools relay /raspicam_node/image/compressed /kamera
	# 

	recebedor = rospy.Subscriber(topico_imagem, CompressedImage, roda_todo_frame, queue_size=4, buff_size = 2**24)
	print("Usando ", topico_imagem)

	velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 1)
	recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)

	try:
		alinhamento = False
		achou = False

		while not rospy.is_shutdown():
			achoucreeper(media, centro,medida, alinhamento,achou)

	except rospy.ROSInterruptException:
	    print("Ocorreu uma exceção com o rospy")
