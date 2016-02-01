

#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv\cxcore.h>
#include "opencv\cv.h"
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace cv;
using namespace std;

// Create memory for calculations
static CvMemStorage* storage0 = 0;
static CvMemStorage* storage1 = 0;
static CvMemStorage* storage2 = 0;

// Create a new Haar classifier
static CvHaarClassifierCascade* cascade = 0;


int main()
{
	int numBoards = 10;  //define el n�mero de tableros de ajedrez que se va a capturar
	int numCornersHor=7; //n�mero de esquinas internas a lo ancho del tablero (pts_imgeto patr�n de calibraci�n)
	int numCornersVer=7; //n�mero de esquinas internas a lo largo del tablero (pts_imgeto patr�n de calibraci�n)
	char distortedImage[100];

	VideoCapture vcap;
	Mat frame;
	const std::string video = "rtsp://200.126.19.98//Streaming/Channels/0";

	if (!vcap.open(video)) {
		std::cout << "Error opening video stream or file" << std::endl;
		return -1;
	}
	int frame_width = vcap.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = vcap.get(CV_CAP_PROP_FRAME_HEIGHT);

	//variables adicionales que se usar�n mas adelante
	int numTotalCorners = numCornersHor * numCornersVer;
	Size board_sz = Size(numCornersHor, numCornersVer);

	vector<vector<Point3f>> coord3D;
	vector<vector<Point2f>> coord2D;

	vector<Point2f> corners; //lista temporal de las esquinas internas de los cuadrados del tablero para la imagen actual.
	int successes = 0; //successes es una 'bandera' que me indicar� si un tablero ha sido detetado en la imagen.

	//creamos dos im�genes a usarse durante el proceso de calibraci�n
	Mat image;
	Mat gray_image;

	//pts_img contiene una lista de puntos que corresponde a las coordenadas 3D de todos las esquinas que se repiten para cada una de las im�genes.
	vector<Point3f> pts_img;

	//tomamos las medidas del cuadrado del tablero, lo que va de esquina a esquina (ladoy, ladox) en este caso en mil�metros.
	const float ladox = 30.0;
	const float ladoy = 30.0;

	//inicializa la lista de puntos pts_img. Se considera que el tablero est� en el plano Z=0.
	for (int i = 0; i < numCornersVer; i++) {
		for (int j = 0; j < numCornersHor; j++) {
			pts_img.push_back(Point3f(float(j*ladox), float(i*ladoy), 0.0f));
		}
	}
	

	//////////////////// Bloque principal del algoritmo de calibraci�n ///////////////////////
	//for (;;) {
	//	/*if (!vcap.read(image)) {
	//		std::cout << "No frame" << std::endl;
	//		cvWaitKey();
	//	}*/
	vcap >> image;
		while (successes < numBoards) //permanecemos en el lazo siempre y cuando el n�mero de tableros que la c�mara observe sea menor al n�mero de tablero ingresado
		{

			cvtColor(image, gray_image, CV_BGR2GRAY); //convertimos la captura en escala de grises
			bool found = findChessboardCorners(gray_image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

			if (found) //si encuentra las esquinas en la imagen
			{
				cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));

				drawChessboardCorners(gray_image, board_sz, corners, found);
				drawChessboardCorners(image, board_sz, corners, found); //dibuja las esquinas en la imagen de grises
			}

			//mostramos las esquinas detectadas por pantalla
			imshow("Imagen Original 1", image);
			imshow("Imagen Gris", gray_image);

			//capture >> image; //capturamos el siguiente frame y se repite el proceso
			vcap >> image;
			int key = waitKey(1); //esperamos una tecla

			if (key == 27) //si es ESC salimos del programa
				return 0;

			if (key == ' ' && found != 0) //si es espacio o alcanz� un numero suficiente de frames, almacenamos las esquinas en la lista y procedemos a la calibraci�n
			{
				coord2D.push_back(corners);
				coord3D.push_back(pts_img);
				printf("Snap %d stored of %d! \n", successes + 1, numBoards);

				successes++;

				if (successes >= numBoards)
					break;
			}
		}
		//cv::imshow("Output Window", image);
		//if (cvWaitKey(1) >= 0) break;
	//}
	image.release();
	vcap.release();
	

	////////////////////////////////////
	//ETAPA 2: Proceso de Calibraci�n///
	////////////////////////////////////

	//declaramos las variables que almacenar�n los par�metros intr�nsecos: cameraMatrix y los coeficientes de distorsi�n (distCoeffs).
	Mat cameraMatrix = Mat(3, 3, CV_32FC1);
	Mat distCoeffs;

	//declaramos las variables que almacenar�n los par�metros extr�nsecos: vectores de rotaci�n y traslaci�n respectivamente (rvecs, tvecs).
	vector<Mat> rvecs; //vectores de rotaci�n
	vector<Mat> tvecs; //vectores de traslaci�n

	// los valores (0,0) y (1,1) representan la distancia focal a lo largo del eje X y Y respectivamente
	cameraMatrix.ptr<float>(0)[0] = 1;
	cameraMatrix.ptr<float>(1)[1] = 1;

	//Esta funci�n permite la calibraci�n de la c�mara. Recibe como parametros las coordenadas 3D y 2D previamente encontradas,
	//y adem�s recibe la matriz cameraMatrix que guardar� los valores intrinsecos y la matriz distCoeffs que guardar� los coeficientes de distorsi�n.
	//Despu�s de la ejecuci�n de esta funci�n se obtendr�n dichas matrices llenas, as� como los vectores rvecs, tvecs de rotaci�n y traslaci�n respectivamente.
	double rms = cv::calibrateCamera(coord3D, coord2D, image.size(), cameraMatrix, distCoeffs, rvecs, tvecs);

	//declaro la variable en donde se almacenar� la imagen sin distorsi�n

	Mat imageUndistorted2;
	Mat imageUndistorted;

	Mat image2 = imread(distortedImage, CV_LOAD_IMAGE_COLOR);
	undistort(image2, imageUndistorted2, cameraMatrix, distCoeffs);
	//finalmente mostramos las im�genes: original y calibrada.
	imshow("Imagen Distorsionada", image2);
	imshow("Imagen sin Distorsi�n", imageUndistorted2);


	while (1)
	{
		//capture >> image;

		//una vez que tenemos los coeficientes de distorsi�n, se usa la funci�n "undistort" para calibrar las distorsiones en la imagen.
		undistort(image, imageUndistorted, cameraMatrix, distCoeffs);

		//finalmente mostramos las im�genes: original y calibrada.
		//imshow("Imagen C�mara Distorsionada", image);
		// imshow("Imagen C�mara sin Distorsi�n", imageUndistorted);
		waitKey(0);
	}

	waitKey(0);
	//capture.release();

	return 0;
}
