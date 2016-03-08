#include <cv.h>
#include <cxcore.h> 
#include <highgui.h> 
#include <iostream>
using namespace std;
// the minimum object size
int min_face_height = 50;
int min_face_width = 50;
int main( int argc , char ** argv ){
    string image_name="lena.bmp";
    // Load image
    IplImage* image_detect=cvLoadImage(image_name.c_str(), 1);
    string cascade_name="C:/OpenCV2.0/data/haarcascades/haarcascade_frontalface_alt.xml";
    
    // Load cascade
    CvHaarClassifierCascade* classifier=(CvHaarClassifierCascade*)cvLoad(cascade_name.c_str(), 0, 0, 0);
    if(!classifier){
        cerr<<"ERROR: Could not load classifier cascade."<<endl;
         system("pause");
        return -1;
    }
    CvMemStorage* facesMemStorage=cvCreateMemStorage(0);
    IplImage* tempFrame=cvCreateImage(cvSize(image_detect->width, image_detect->height), IPL_DEPTH_8U, image_detect->nChannels);
    if(image_detect->origin==IPL_ORIGIN_TL){
        cvCopy(image_detect, tempFrame, 0);    }
    else{
        cvFlip(image_detect, tempFrame, 0);    }
    cvClearMemStorage(facesMemStorage);
CvSeq* faces=cvHaarDetectObjects(tempFrame, classifier, facesMemStorage, 1.1, 3
, CV_HAAR_DO_CANNY_PRUNING, cvSize(min_face_width, min_face_height));
    if(faces){
        for(int i=0; i<faces->total; ++i){
            // Setup two points that define the extremes of the rectangle,
            // then draw it to the image
            CvPoint point1, point2;
            CvRect* rectangle = (CvRect*)cvGetSeqElem(faces, i);
            point1.x = rectangle->x;
            point2.x = rectangle->x + rectangle->width;
            point1.y = rectangle->y;
            point2.y = rectangle->y + rectangle->height;
            cvRectangle(tempFrame, point1, point2, CV_RGB(255,0,0), 3, 8, 0);
        }
    }
    // Save the image to a file
    cvSaveImage("02.bmp", tempFrame);
    // Show the result in the window
    cvNamedWindow("Face Detection Result", 1);
    cvShowImage("Face Detection Result", tempFrame);
    cvWaitKey(0);
    cvDestroyWindow("Face Detection Result");
    // Clean up allocated OpenCV objects
    cvReleaseMemStorage(&facesMemStorage);
    cvReleaseImage(&tempFrame);
    cvReleaseHaarClassifierCascade(&classifier);
    cvReleaseImage(&image_detect);
    system("pause");
    return EXIT_SUCCESS;
}
