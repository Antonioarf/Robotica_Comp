#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rospy

import numpy as np

from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan

medida = -1
def scaneou(dado):
	print("Faixa valida: ", dado.range_min , " - ", dado.range_max )
	print("Leituras:")
	global medida
	medida = np.array(dado.ranges).round(decimals=2)[0]
	print(np.array(dado.ranges).round(decimals=2))
	#print("Intensities")
	#print(np.array(dado.intensities).round(decimals=2))

	


if __name__=="__main__":

	rospy.init_node("le_scan")

	velocidade_saida = rospy.Publisher("/cmd_vel", Twist, queue_size = 3 )
	recebe_scan = rospy.Subscriber("/scan", LaserScan, scaneou)



	while not rospy.is_shutdown():
		if medida <= 1:
			velocidade = Twist(Vector3(-0.2, 0, 0), Vector3(0, 0, 0))
			velocidade_saida.publish(velocidade)
			rospy.sleep(2)
		else:
			velocidade = Twist(Vector3(0.2, 0, 0), Vector3(0, 0, 0))
			velocidade_saida.publish(velocidade)
			rospy.sleep(2)


