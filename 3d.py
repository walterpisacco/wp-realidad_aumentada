import cv2
import numpy as np
from calibracion.calibracion import *

# inicializamos el detector de arucos
parametros = cv2.aruco.DetectorParameters_create()

#cargamos el diccionario de nuestro aruco 5x5 (25 bits) 
diccionario =  cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)

cap = cv2.VideoCapture(0)
cap.set(3,1920) #definimos el alto y ancho de la imagen
cap.set(4,1080)
cont = 0

#calibracion
calibracion = calibracion()
matrix, dist = calibracion.calibracion_cam()

while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	esquinas,ids, candidatos_malos = cv2.aruco.detectMarkers(gray, diccionario, parameters=parametros)
	try:
		#si hay marcadores detectados por el detector
		if np.all(ids != None):
			#iteramos en marcadores
			for i in range(0, len(ids)):
				#estima la posicion de cada marcador y devuelve los valores rvec y tvec --- diferentes de los coeficientes
				rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(esquinas[i],0.02, matrix, dist)

				#eliminamos el error de la matriz de valores numpy
				(rvec - tvec).any()

				#dibujamos un cuadrado alrededor de los marcadores
				cv2.aruco.drawDetectedMarkers(frame,esquinas)

				#dibujamos ejes

				#cv2.aruco.drawAxis(frame, matrix, dist, rvec, tvec, 0.01)

				#coordenadas x del centro del marcador
				c_x = (esquinas[i][0][0][0] + esquinas[i][0][1][0] + esquinas[i][0][2][0] + esquinas[i][0][3][0]) /4

				#coordenadas y del centro del marcador
				c_y = (esquinas[i][0][0][1] + esquinas[i][0][1][1] + esquinas[i][0][2][1] + esquinas[i][0][3][1]) /4				


				#extraemos los puntos de las esquinas en coordenadas separadas
				c1 = (esquinas[0][0][0][0], esquinas[0][0][0][1])
				c2 = (esquinas[0][0][1][0], esquinas[0][0][1][1])
				c3 = (esquinas[0][0][2][0], esquinas[0][0][2][1])
				c4 = (esquinas[0][0][3][0], esquinas[0][0][3][1])
				v1, v2 = c1[0],c1[1]
				v3, v4 = c2[0],c2[1]
				v5, v6 = c3[0],c3[1]
				v7, v8 = c4[0],c4[1]

				#dibujamos cubo
				#cara interior
				cv2.line(frame, (int(v1), int(v2)), (int(v3), int(v4)), (255,255,0), 3)
				cv2.line(frame, (int(v5), int(v6)), (int(v7), int(v8)), (255,255,0), 3)
				cv2.line(frame, (int(v1), int(v2)), (int(v7), int(v8)), (255,255,0), 3)
				cv2.line(frame, (int(v3), int(v4)), (int(v5), int(v6)), (255,255,0), 3)
				#cara superior
				cv2.line(frame, (int(v1), int(v2 - 200)), (int(v3), int(v4 - 200)), (255,255,0), 3)
				cv2.line(frame, (int(v5), int(v6 - 200)), (int(v7), int(v8 - 200)), (255,255,0), 3)
				cv2.line(frame, (int(v1), int(v2 - 200)), (int(v7), int(v8 - 200)), (255,255,0), 3)
				cv2.line(frame, (int(v3), int(v4 - 200)), (int(v5), int(v6 - 200)), (255,255,0), 3)				
				#caras laterales
				cv2.line(frame, (int(v1), int(v2 - 200)), (int(v1), int(v2)), (255,255,0), 3)
				cv2.line(frame, (int(v3), int(v4 - 200)), (int(v3), int(v4)), (255,255,0), 3)
				cv2.line(frame, (int(v5), int(v6 - 200)), (int(v5), int(v6)), (255,255,0), 3)
				cv2.line(frame, (int(v7), int(v8 - 200)), (int(v7), int(v8)), (255,255,0), 3)	

	except NameError:
		if ids is None or len(ids) == 0:
			print("**** Falla l detectar marker ****")
	#	else:
	#		print(NameError)
	
	cv2.namedWindow ("Realidad", 0) #CV_WINDOW_NORMAL es 0
	cv2.imshow("Realidad", frame)

	k = cv2.waitKey(1)

	if k == 97:
		print("Imagen guardada")
		cv2.imwrite("cali{}.png".format(cont),frame)
		cont = cont + 1

	if k == 27:
		break

cap.release()
cv2.destroyAllWindows()