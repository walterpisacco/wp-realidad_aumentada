import cv2
import numpy as np

# inicializamos el detector de arucos
parametros = cv2.aruco.DetectorParameters_create()

#cargamos el diccionario de nuestro aruco 5x5 (25 bits) 
diccionario =  cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_100)

cap = cv2.VideoCapture(0)
cap.set(3,1920) #definimos el alto y ancho de la imagen
cap.set(4,1080)

cont = 0 #lo uso para sacar fotos

while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	esquinas,ids, candidatos_malos = cv2.aruco.detectMarkers(gray, diccionario, parameters=parametros)

	if np.all(ids != None):
		aruco = cv2.aruco.drawDetectedMarkers(frame, esquinas)

		c1 = (esquinas[0][0][0][0], esquinas[0][0][0][1])
		c2 = (esquinas[0][0][1][0], esquinas[0][0][1][1])
		c3 = (esquinas[0][0][2][0], esquinas[0][0][2][1])
		c4 = (esquinas[0][0][3][0], esquinas[0][0][3][1])

		copy = frame
		imagen = cv2.imread("foto2.jpg")
		tamanio = imagen.shape
		puntos_aruco = np.array([c1,c2,c3,c4])

		#organizamos las coordenadas de la imagen en otra matriz
		puntos_imagen = np.array([
			[0,0],
			[tamanio[1] - 1,0],
			[tamanio[1] - 1, tamanio[1] - 1],
			[0, tamanio[0] - 1]
			], dtype = float)

		#realizamos la superposicion de imagenes (Homografia)
		h, estado = cv2.findHomography(puntos_imagen, puntos_aruco)

		#realizamos la transformacion de perspesctiva
		perspectiva = cv2.warpPerspective(imagen, h, (copy.shape[1], copy.shape[0]))
		cv2.fillConvexPoly(copy, puntos_aruco.astype(int),0 ,16)
		copy = copy + perspectiva
		cv2.imshow("Realidad", copy)

	else:
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