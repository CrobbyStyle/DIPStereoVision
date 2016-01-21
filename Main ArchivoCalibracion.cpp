 //*****************************************************************//
 //            Como calibrar una cámara en OpenCv				    // 
 //   Calibración de cámara a partir de un patrón de calibración    //
 //          Nombre: Dr. Boris Vintimilla B.   					    //
 //*****************************************************************//

//En este laboratorio se aprenderá el proceso para la calibración de una cámara dado 
//un patrón de calibración. El patrón de calibración a usar es un tablero de ajedrez, 
//las librerías de OpenCv ofrecen funciones que permiten este proceso dado este tipo  
//de patrón de calibración (tablero de ajedrez). El código esta documentado de tal forma
//que el usuario pueda comprender el proceso seguido y poner en práctica usando un tablero 
//de ajedrez a través de una señal de video, visualice la detección de esquinas, y los
//resultados de la imagen calibrada.

//El codigo de este program usa como referencia bibliografica el siguiente enlace: http://aishack.in/tutorials/calibrating-undistorting-opencv-oh-yeah/

//librerias usadas
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <vector>
#include <stdlib.h>
#include <stdio.h>


using namespace cv;
using namespace std;

int main()
{
int numBoards = 0;  //define el número de tableros de ajedrez que se va a capturar
int numCornersHor; //número de esquinas internas a lo ancho del tablero (pts_imgeto patrón de calibración)
int numCornersVer; //número de esquinas internas a lo largo del tablero (pts_imgeto patrón de calibración)
char distortedImage[100];

//obtenemos estos valores por teclado
printf("Enter number of inner corners along width: ");
scanf("%d", &numCornersHor);
printf("Enter number of inner corners along height: ");
scanf("%d", &numCornersVer);
printf("Enter the number of images for calibration: ");
scanf("%d", &numBoards);
printf("Enter the name of distorted image: ");
scanf("%s", &distortedImage);

//variables adicionales que se usarán mas adelante
int numTotalCorners = numCornersHor * numCornersVer; 
Size board_sz = Size(numCornersHor, numCornersVer);


VideoCapture capture = VideoCapture(0); //captura las imágenes con el patrón de calibración (tablero de ajedrez) desde una cámara
//VideoCapture capture = VideoCapture("tablero.jpg"); //captura las imágenes con el patrón de calibración (tablero de ajedrez) desde un archivo de imagen

//Para cada imagen creamos una lista que contiene una lista de puntos-3D de la escena del mundo real.
//coord3D contendrá la posición 3D de los puntos que representa a las esquinas internas de los cuadros del tablero en la escena 3D del mundo real.
vector<vector<Point3f>> coord3D; 

//Para cada imagen creamos una lista que contiene una lista de puntos-2D en la imagen
//coord2D contendrá la posición 2D de los puntos que representan a las esquinas internas de los cuadros del tablero en la imagen 2D.
vector<vector<Point2f>> coord2D; 

//Una vez que se tienen las posiciones 3D y las posiciones 2D de las esquinas internas del tablero se puede calcular la relación entre ambas
//NOTA: dado que usamos un tablero de ajedrez estos puntos tienen una relación definida entre ellos, esto es:
//la posición de los puntos está sobre las lineas y los cuadrados del tablero.
//entonces la relación (esperado-real) puede ser usada para corregir la distorsión de la imagen.

vector<Point2f> corners; //lista temporal de las esquinas internas de los cuadrados del tablero para la imagen actual.
int successes=0; //successes es una 'bandera' que me indicará si un tablero ha sido detetado en la imagen.

//creamos dos imágenes a usarse durante el proceso de calibración
Mat image;
Mat gray_image;

//asigna cada frame de la señal de video a la variable image (imagen a color).
capture >> image;

//pts_img contiene una lista de puntos que corresponde a las coordenadas 3D de todos las esquinas que se repiten para cada una de las imágenes.
vector<Point3f> pts_img;

//tomamos las medidas del cuadrado del tablero, lo que va de esquina a esquina (ladoy, ladox) en este caso en milímetros.
const float ladox = 30.0;
const float ladoy = 30.0;

//inicializa la lista de puntos pts_img. Se considera que el tablero está en el plano Z=0.
for( int i = 0; i < numCornersVer; i++ ) {
	for( int j = 0; j < numCornersHor; j++ ) {
		pts_img.push_back( Point3f(float( j*ladox ), float( i*ladoy ), 0.0f ) );
	}
}
//for(int j=0; j<numTotalCorners; j++)
    //pts_img.push_back( Point3f(float(j/numCornersHor), float(j%numCornersHor), 0.0f ) ); 

//////////////////// Bloque principal del algoritmo de calibración ///////////////////////

while( successes < numBoards ) //permanecemos en el lazo siempre y cuando el número de tableros que la cámara observe sea menor al número de tablero ingresado
{ 
    cvtColor(image, gray_image, CV_BGR2GRAY); //convertimos la captura en escala de grises

//El algoritmo tiene 2 etapas: 
//Etapa 1: detecta las esquinas del tablero.
//Etapa 2: proceso de calibración.

////////////////////////////////////////////////////////////////////////////////////////////
//  ETAPA 1: detecta las esquinas del tablero y sus correspondientes coordenadas 2D y 3D ///
////////////////////////////////////////////////////////////////////////////////////////////

    // findChessboardCorners detecta el número de esquinas en el tablero de la imagen y lo compara con el board_sz previamente definido.
    // Si estos valores son iguales devuelve 1 caso contrario 0.
	// si las esquinas fueron detectadas almacena sus coordenadas en 'corners'.
    bool found = findChessboardCorners(gray_image, board_sz, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

    if(found) //si encuentra las esquinas en la imagen
    {
		//se calculan los pixeles vecinos que conforman las esquinas con el propósito de mejorar la precisión de la calibración (esto se lo hace desde la imagen de grises) 
		//estos píxeles vecinos son llamados sub-píxeles de la esquina detectada
		//Size define el tamaño del vecindario (sub-píxeles) de la esquina detectada, (-1,-1) es para no definirla.
		//TermCriteria ayuda a establecer parametros como: cuantas iteraciones realiza, cantidad de error permitido.
        cornerSubPix(gray_image, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));


		//dibuja las esquinas detectadas en una imagen de grises
		drawChessboardCorners(gray_image, board_sz, corners, found); 		
		drawChessboardCorners(image, board_sz, corners, found); //dibuja las esquinas en la imagen de grises
	}

	//mostramos las esquinas detectadas por pantalla
    imshow("Imagen Original 1", image);
    imshow("Imagen Gris", gray_image);
	
    capture >> image; //capturamos el siguiente frame y se repite el proceso

    int key = waitKey(1); //esperamos una tecla

    if(key==27) //si es ESC salimos del programa
        return 0;

    if(key==' ' && found!=0) //si es espacio o alcanzó un numero suficiente de frames, almacenamos las esquinas en la lista y procedemos a la calibración
	{
        coord2D.push_back(corners);
        coord3D.push_back(pts_img);
        printf("Snap %d stored of %d! \n", successes+1, numBoards);

        successes++;

        if(successes>=numBoards)
            break;
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

Mat imageUndistorted2; 
Mat imageUndistorted; 

Mat image2 = imread(distortedImage,CV_LOAD_IMAGE_COLOR);
undistort(image2, imageUndistorted2, cameraMatrix, distCoeffs);
//finalmente mostramos las imágenes: original y calibrada.
    imshow("Imagen Distorsionada", image2);
    imshow("Imagen sin Distorsión", imageUndistorted2);


while(1)
{
    capture >> image;

	//una vez que tenemos los coeficientes de distorsión, se usa la función "undistort" para calibrar las distorsiones en la imagen.
    undistort(image, imageUndistorted, cameraMatrix, distCoeffs);

	//finalmente mostramos las imágenes: original y calibrada.
    //imshow("Imagen Cámara Distorsionada", image);
   // imshow("Imagen Cámara sin Distorsión", imageUndistorted);
    waitKey(1);
}

waitKey(1);
capture.release();

return 0;
}