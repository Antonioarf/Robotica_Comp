#! /usr/bin/env python
# -*- coding:utf-8 -*-

import rospy
from geometry_msgs.msg import Twist, Vector3

v = 0.05 # Velocidade linear
w = 0.196  # Velocidade angular

if __name__ == "__main__":
    rospy.init_node("roda_exemplo")
    pub = rospy.Publisher("cmd_vel", Twist, queue_size=3)

    try:
        while not rospy.is_shutdown():
			vel = Twist(Vector3(0,0,0), Vector3(0,0,w))
			pub.publish(vel)
			rospy.sleep(8.0)
			vel = Twist(Vector3(0,0,0), Vector3(0,0,0))
			pub.publish(vel)
			rospy.sleep(4.0)
			vel2 = Twist(Vector3(0.02,0,0), Vector3(0,0,0))
			pub.publish(vel2)
			rospy.sleep(3.0)
			vel2 = Twist(Vector3(0.15,0,0), Vector3(0,0,0))
			pub.publish(vel2)
			rospy.sleep(2.0)
			vel2 = Twist(Vector3(0.05,0,0), Vector3(0,0,0))
			pub.publish(vel2)
			rospy.sleep(5.0)
            
    except rospy.ROSInterruptException:
        print("Ocorreu uma exceção com o rospy")