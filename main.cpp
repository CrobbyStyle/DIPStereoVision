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
	int numBoards = 9;  //define el número de tableros de ajedrez que se va a capturar
	int numCornersHor = 7; //número de esquinas internas a lo ancho del tablero (pts_imgeto patrón de calibración)
	int numCornersVer = 7; //número de esquinas internas a lo largo del tablero (pts_imgeto patrón de calibración)
	char distortedImage[100];

	VideoCapture vcap, vcap2;
	Mat frame, frame2;
	const std::string videoR = "http://200.126.19.123/cgi-bin/mjpg?stream=0?jiggly.mjpg";
	const std::string videoL = "http://200.126.19.98/cgi-bin/mjpg?stream=0?jiggly.mjpg";
	if (!vcap.open(videoR)) {
		std::cout << "Error opening video stream or file" << std::endl;
		return -1;
	}
	if (!vcap2.open(videoL)) {
		std::cout << "Error opening video stream or file" << std::endl;
		return -1;
	}

	vcap.set(CV_CAP_PROP_FPS, 30);

	int frame_width = vcap.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = vcap.get(CV_CAP_PROP_FRAME_HEIGHT);
	
	int frame_width2 = vcap2.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height2 = vcap2.get(CV_CAP_PROP_FRAME_HEIGHT);
	//variables adicionales que se usarán mas adelante
	int numTotalCorners = numCornersHor * numCornersVer;
	Size board_sz = Size(numCornersHor, numCornersVer);

	vector<vector<Point3f>> coord3D;
	vector<vector<Point2f>> coord2D;

	vector<Point2f> corners; //lista temporal de las esquinas internas de los cuadrados del tablero para la imagen actual.
	int successes = 0; //successes es una 'bandera' que me indicará si un tablero ha sido detetado en la imagen.

	//creamos dos imágenes a usarse durante el proceso de calibración
	Mat image, image2;
	Mat gray_image;

	//pts_img contiene una lista de puntos que corresponde a las coordenadas 3D de todos las esquinas que se repiten para cada una de las imágenes.
	vector<Point3f> pts_img;

	//tomamos las medidas del cuadrado del tablero, lo que va de esquina a esquina (ladoy, ladox) en este caso en milímetros.
	const float ladox = 30.0;
	const float ladoy = 30.0;

	//inicializa la lista de puntos pts_img. Se considera que el tablero está en el plano Z=0.
	for (int i = 0; i < numCornersVer; i++) {
		for (int j = 0; j < numCornersHor; j++) {
			pts_img.push_back(Point3f(float(j*ladox), float(i*ladoy), 0.0f));
		}
	}


	//////////////////// Bloque principal del algoritmo de calibración ///////////////////////
	
	for (;;) {
		vcap >> image;
		if (successes < numBoards) //permanecemos en el lazo siempre y cuando el número de tableros que la cámara observe sea menor al número de tablero ingresado
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

			vcap >> image; //capturamos el siguiente frame y se repite el proceso
			//vcap.read(image);
			int key = waitKey(1); //esperamos una tecla

			if (key == 27) //si es ESC salimos del programa
				return 0;

			if (key == ' ' && found != 0) //si es espacio o alcanzó un numero suficiente de frames, almacenamos las esquinas en la lista y procedemos a la calibración
			{
				coord2D.push_back(corners);
				coord3D.push_back(pts_img);
				printf("Snap %d stored of %d! \n", successes + 1, numBoards);

				successes++;

				if (successes >= numBoards)
					break;
			}
		}
	}


	////////////////////////////////////
	//ETAPA 2: Proceso de Calibración///
	////////////////////////////////////

	//declaramos las variables que almacenarán los parámetros intrínsecos: cameraMatrix y los coeficientes de distorsión (distCoeffs).
	Mat cameraMatrix = Mat(3, 3, CV_32FC1);
	Mat distCoeffs;

	//declaramos las variables que almacenarán los parámetros extrínsecos: vectores de rotación y traslación respectivamente (rvecs, tvecs).
	vector<Mat> rvecs; //vectores de rotación
	vector<Mat> tvecs; //vectores de traslación

	// los valores (0,0) y (1,1) representan la distancia focal a lo largo del eje X y Y respectivamente
	cameraMatrix.ptr<float>(0)[0] = 1;
	cameraMatrix.ptr<float>(1)[1] = 1;

	//Esta función permite la calibración de la cámara. Recibe como parametros las coordenadas 3D y 2D previamente encontradas,
	//y además recibe la matriz cameraMatrix que guardará los valores intrinsecos y la matriz distCoeffs que guardará los coeficientes de distorsión.
	//Después de la ejecución de esta función se obtendrán dichas matrices llenas, así como los vectores rvecs, tvecs de rotación y traslación respectivamente.
	double rms = cv::calibrateCamera(coord3D, coord2D, image.size(), cameraMatrix, distCoeffs, rvecs, tvecs);

	//declaro la variable en donde se almacenará la imagen sin distorsión

	Mat imageUndistorted, imageUndistorted2;
	Mat g1, g2;
	Mat disp, disp8;

	// 30 fps para ambas señales
	vcap.set(CV_CAP_PROP_FPS, 30);
	vcap2.set(CV_CAP_PROP_FPS, 30);
	
	for (;;)
	{
		// extraigo ambas señales
		vcap.read(image);
		vcap2.read(image2);

		// corrijo efecto barril en ambas señales
		undistort(image, imageUndistorted, cameraMatrix, distCoeffs);
		undistort(image2, imageUndistorted2, cameraMatrix, distCoeffs);

		// gris de ambas señales sin distorcion
		cvtColor(imageUndistorted, g1, CV_BGR2GRAY);
		cvtColor(imageUndistorted2, g2, CV_BGR2GRAY);
		
		// inicializando variables del mapa de disparidad
		StereoBM sbm;
		sbm.state->SADWindowSize = 9;
		sbm.state->numberOfDisparities = 112;
		sbm.state->preFilterSize = 5;
		sbm.state->preFilterCap = 61;
		sbm.state->minDisparity = -39;
		sbm.state->textureThreshold = 507;
		sbm.state->uniquenessRatio = 0;
		sbm.state->speckleWindowSize = 0;
		sbm.state->speckleRange = 8;
		sbm.state->disp12MaxDiff = 1;

		// calculo mapa de disparidad entre señal derecha e izquierda
		sbm(g1, g2, disp);
		normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
	
		//finalmente mostramos las imágenes: original y calibrada.
		imshow("Imagen R Cámara Distorsionada", image);
		imshow("Imagen R Cámara sin Distorsión", imageUndistorted);
		imshow("Imagen L Cámara Distorsionada", image2);
		imshow("Imagen L Cámara sin Distorsión", imageUndistorted2);

		// mostramos mapa de disparidad
		imshow("disp", disp8);

		waitKey(1);
	}

	waitKey(0);
	vcap.release();
	vcap2.release();
	image.release();
	image2.release();
	return 0;
}

//void main(){
//	Mat img1, img2, g1, g2;
//	Mat disp, disp8;
//
//	img1 = imread("leftImage.jpg");
//	img2 = imread("rightImage.jpg");
//
//	cvtColor(img1, g1, CV_BGR2GRAY);
//	cvtColor(img2, g2, CV_BGR2GRAY);
//
//	StereoBM sbm;
//	sbm.state->SADWindowSize = 9;
//	sbm.state->numberOfDisparities = 112;
//	sbm.state->preFilterSize = 5;
//	sbm.state->preFilterCap = 61;
//	sbm.state->minDisparity = -39;
//	sbm.state->textureThreshold = 507;
//	sbm.state->uniquenessRatio = 0;
//	sbm.state->speckleWindowSize = 0;
//	sbm.state->speckleRange = 8;
//	sbm.state->disp12MaxDiff = 1;
//
//	sbm(g1, g2, disp);
//	normalize(disp, disp8, 0, 255, CV_MINMAX, CV_8U);
//
//	imshow("left", img1);
//	imshow("right", img2);
//	imshow("disp", disp8);
//	cvWaitKey(0);
//}
