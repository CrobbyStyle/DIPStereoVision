
//**********************************************************//
//    D E T E C C I O N   Y  R E C O N O C I M I E N T O	//
//					D E    M O V I M I E N T O              // 
//	  D E  U N A   C A M A R A   D E   V I G I L A N C I A  //
//    dato de entrada: Señal de video de una camara IP      // 
//															//
//            Nombre: Jose Velez y Marlon Espinoza	        //
//**********************************************************//

#include <opencv2\opencv.hpp>
#include <opencv\cxcore.h>
#include "opencv\cv.h"
#include "opencv\highgui.h"
using namespace cv;

// Create memory for calculations
static CvMemStorage* storage0 = 0;
static CvMemStorage* storage1 = 0;
static CvMemStorage* storage2 = 0;

// Create a new Haar classifier
static CvHaarClassifierCascade* cascade = 0;

// Function prototype for detecting and drawing an object from an image
void detect_and_draw(Mat img);

// Create a string that contains the cascade name
const char* cascade_name = "haarcascade_frontalface_alt.xml";


/*    "haarcascade_profileface.xml";*/

int eye_opt = 0;//Variable que indica si se activa efecto ojos o no
int rect_opt = 0;//Varaible que indica si se activa efecto rostro o no
int boca_opt = 0;//Varaible que indica si se activa efecto boca o no
int nariz_opt = 0;//Varaible que indica si se activa efecto nariz o no


int main(int argc, char **argv)
{
	VideoCapture vcap;
	Mat frame;
	Mat copy;
	//const std::string video = "rtsp://camcidis1:us3rC4m1@200.126.19.101:554//Streaming/Channels/1";
	//200.126.19.82
	const std::string video = "rtsp://200.126.19.82//Streaming/Channels/0";
	//cvNamedWindow("result", 1);
	// Carga el HaarClassifierCascade
	cascade = (CvHaarClassifierCascade*)cvLoad(cascade_name, 0, 0, 0);

	// Se reserva memoria para el almacenamiento de las caracteristicas a buscar con el archivo cascade
	storage0 = cvCreateMemStorage(0);
	storage1 = cvCreateMemStorage(0);
	storage2 = cvCreateMemStorage(0);

	if (!vcap.open(video)) {
		std::cout << "Error opening video stream or file" << std::endl;
		return -1;
	}


	for (;;) {
		if (!vcap.read(frame)) {
			std::cout << "No frame" << std::endl;
			cvWaitKey();
		}
		copy = frame.clone();

		detect_and_draw(frame);
		cv::imshow("Output Window", frame);
		if (cvWaitKey(1) >= 0) break;
	}
	frame.release();
	vcap.release();
	cvDestroyWindow("result");
}
// Funcion para detectar y dibujar ciertas caracteristicas dentro de una imagen
void detect_and_draw(Mat img)
{
	int scale = 1;

	// Se crea una nueva imagen basada en la de origen
	IplImage* temp = cvCloneImage(&(IplImage)img);

	// Se crean 2 puntos que nos indicaran el origen del rostro
	CvPoint pt1, pt2;
	int i;

	// Limpiamos la memoria que ya fue usada
	cvClearMemStorage(storage0);
	cvClearMemStorage(storage1);
	cvClearMemStorage(storage2);

	// Se verifica que el archivo Cascade ya este cargado
	if (cascade)
	{

		// Pueden haber mas de una cara en una imagen, entonces se procede 
		//a crear una secuencia de imagenes
		CvSeq* faces = cvHaarDetectObjects(temp, cascade, storage0, 1.1, 2, CV_HAAR_DO_CANNY_PRUNING, cvSize(40, 40));


		// Lazo donde se manejan las caras 
		for (i = 0; i < (faces ? faces->total : 0); i++)
		{

			// Create a new rectangle for drawing the face
			CvRect* r = (CvRect*)cvGetSeqElem(faces, i);

			// Encuentra las dimensiones de las caras y las escala si es necesario
			pt1.x = r->x*scale;
			pt2.x = (r->x + r->width)*scale;
			pt1.y = r->y*scale;
			pt2.y = (r->y + r->height)*scale;
			int ancho = abs(pt2.x - pt1.x);
			int alto = abs(pt2.y - pt1.y);
			int resolucion = (ancho + alto) / 8;

			cvRectangle(temp, pt1, pt2, CV_RGB(255, 0, 0), 3, 8, 0);

		}

	}

	//cv::imshow("Output Window", temp);
	//cvShowImage("result", temp);


	cvReleaseImage(&temp);
}

