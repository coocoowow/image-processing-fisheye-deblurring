#define _CRT_SECURE_NO_DEPRECATE //去除fopen報警告
#include <stdio.h> 
#include <cv.h> 
#include <highgui.h> 
#include <iostream>
#include <fstream>
#include <string>
using namespace cv;
using namespace std;
#define kappa 10000




void sampleImage(const IplImage* arr, float idx0, float idx1, CvScalar& res){
  if(idx0<0 || idx1<0 || idx0>(cvGetSize(arr).height-1) || idx1>(cvGetSize(arr).width-1)){
    res.val[0]=0;
    res.val[1]=0;
    res.val[2]=0;
    res.val[3]=0;
    return;
  }
  float idx0_fl=floor(idx0);
  float idx0_cl=ceil(idx0);
  float idx1_fl=floor(idx1);
  float idx1_cl=ceil(idx1);
 
  CvScalar s1=cvGet2D(arr,(int)idx0_fl,(int)idx1_fl);
  CvScalar s2=cvGet2D(arr,(int)idx0_fl,(int)idx1_cl);
  CvScalar s3=cvGet2D(arr,(int)idx0_cl,(int)idx1_cl);
  CvScalar s4=cvGet2D(arr,(int)idx0_cl,(int)idx1_fl);
  float x = idx0 - idx0_fl;
  float y = idx1 - idx1_fl;
  res.val[0]= s1.val[0]*(1-x)*(1-y) + s2.val[0]*(1-x)*y + s3.val[0]*x*y + s4.val[0]*x*(1-y);
  res.val[1]= s1.val[1]*(1-x)*(1-y) + s2.val[1]*(1-x)*y + s3.val[1]*x*y + s4.val[1]*x*(1-y);
  res.val[2]= s1.val[2]*(1-x)*(1-y) + s2.val[2]*(1-x)*y + s3.val[2]*x*y + s4.val[2]*x*(1-y);
  res.val[3]= s1.val[3]*(1-x)*(1-y) + s2.val[3]*(1-x)*y + s3.val[3]*x*y + s4.val[3]*x*(1-y);
}
float xscale;
float yscale;
float xshift;
float yshift;
float getRadialX(float x,float y,float cx,float cy,float k){
  x = (x*xscale+xshift);
  y = (y*yscale+yshift);
  float res = x+((x-cx)*k*((x-cx)*(x-cx)+(y-cy)*(y-cy)));
  return res;
}
float getRadialY(float x,float y,float cx,float cy,float k){
  x = (x*xscale+xshift);
  y = (y*yscale+yshift);
  float res = y+((y-cy)*k*((x-cx)*(x-cx)+(y-cy)*(y-cy)));
  return res;
}
float thresh = 1;
float calc_shift(float x1,float x2,float cx,float k){
  float x3 = x1+(x2-x1)*0.5;
  float res1 = x1+((x1-cx)*k*((x1-cx)*(x1-cx)));
  float res3 = x3+((x3-cx)*k*((x3-cx)*(x3-cx)));
  //  std::cerr<<"x1: "<<x1<<" - "<<res1<<" x3: "<<x3<<" - "<<res3<<std::endl;
  if(res1>-thresh && res1 < thresh)
    return x1;
  if(res3<0){
    return calc_shift(x3,x2,cx,k);
  }
  else{
    return calc_shift(x1,x3,cx,k);
  }
}



Mat* openRaw(std::string fileName, int sz)
{
	unsigned char *tmpi, *tmpo;
	int size = sz * sz;
	int* sizePtr = &size;

	char FileNameOri[30];
	strcpy(FileNameOri, fileName.c_str());

	CvMat *CVmatTmp = cvCreateMat(sz,sz, CV_8UC1);	

	tmpi = new unsigned char[size]; 
	tmpo = new unsigned char[size];

	FILE *tmp;

	tmp = fopen(FileNameOri,"rb");

	Mat cmMat;
	Mat* cm = new Mat;

	if(tmp == NULL)
	{
		puts("Loading File Error!");
	}
	else
	{
		fread(tmpi,1,size,tmp);
		int m1 = 0;
		int m2 = 0;

		cvSetData(CVmatTmp,tmpi,CVmatTmp->step);

		cmMat.create(sz, sz, CV_8UC1);
		cm[0] = cmMat;	//[0]
		*cm = cvarrToMat(CVmatTmp);
	}

	return cm;
}

void Q1()
{
	Mat oneaMat;
	oneaMat.create(256, 256, CV_8UC1);
	Mat* CB256 = &oneaMat;
	//CB256 = openRaw("chessboard_256.raw", 256);
	CB256 = openRaw("lena_256.raw", 256);

	//IplImage* src = cvLoadImage( argv[1], 1 );
	IplImage* src = &IplImage(*CB256);
	IplImage* dst = cvCreateImage(cvGetSize(src),src->depth,src->nChannels);
	IplImage* dst2 = cvCreateImage(cvGetSize(src),src->depth,src->nChannels);

	//float K=atof(argv[3]);
	//float K = 0.001f;
	float K = 0.001f;

	//float centerX=atoi(argv[4]);
	float centerX = 128;
	//float centerY=atoi(argv[5]);
	float centerY = 128;

	int width = cvGetSize(src).width;
	int height = cvGetSize(src).height;
 
	xshift = calc_shift(0,centerX-1,centerX,K);
	float newcenterX = width-centerX;
	float xshift_2 = calc_shift(0,newcenterX-1,newcenterX,K);
 
	yshift = calc_shift(0,centerY-1,centerY,K);
	float newcenterY = height-centerY;
	float yshift_2 = calc_shift(0,newcenterY-1,newcenterY,K);

	xscale = (width-xshift-xshift_2)/width;
	yscale = (height-yshift-yshift_2)/height;
 
	std::cerr<<xshift<<" "<<yshift<<" "<<xscale<<" "<<yscale<<std::endl;
	std::cerr<<cvGetSize(src).height<<std::endl;
	std::cerr<<cvGetSize(src).width<<std::endl;
 
	for(int j = 0; j < cvGetSize(dst).height; j++)
	{
		for(int i = 0; i < cvGetSize(dst).width; i++)
		{
			CvScalar s;
			float x = getRadialX((float)i,(float)j,centerX,centerY,K);
			float y = getRadialY((float)i,(float)j,centerX,centerY,K);
			sampleImage(src,y,x,s);
			cvSet2D(dst,j,i,s);
		}
	}

#if 0
	cvNamedWindow( "Source1", 1 );
	cvShowImage( "Source1", dst);
	cvWaitKey(0);
#endif
	cvSaveImage("12.jpg", dst, 0);
	cvShowImage( "result", dst);
	cvWaitKey(0);
#if 0
  for(int j=0;j<cvGetSize(src).height;j++)
  {
    for(int i=0;i<cvGetSize(src).width;i++)
	{
		CvScalar s;
		sampleImage(src,j+0.25,i+0.25,s);
		cvSet2D(dst,j,i,s);
    }
  }

  cvNamedWindow( "Source1", 1 );
  cvShowImage( "Source1", src);
  cvWaitKey(0);
#endif
}

void Q1_2()
{
	//	double paramA = -0.7715; 
	//   double paramB = 0.026731; 
	//   double paramC = 0.0; 
	//   double paramD = 1.0 - paramA - paramB - paramC; 
		
	double paramA = -0.36; 
    double paramB = 1.16; 
    double paramC = -1.794; 
    double paramD = 1.0 - paramA - paramB - paramC; 

	Mat oneaMat;
	oneaMat.create(256, 256, CV_8UC1);
	Mat* CB256 = &oneaMat;
	//CB256 = openRaw("lena_256.raw", 256);
	CB256 = openRaw("chessboard_256.raw", 256);
	//Mat copy = CB256->clone();

	Mat result;
	result.create(256, 256, CV_8UC1);

	for (int h = 0; h < CB256->rows; h++)
	{
		for (int w = 0; w < CB256->cols; w++)
		{
			int d = CB256->cols / 2;    // radius of the circle

            // center of dst image
            double centerX = (CB256->cols - 1) / 2.0;
            double centerY = (CB256->cols - 1) / 2.0;

            // cartesian coordinates of the destination point (relative to the centre of the image)
            double deltaX = (w - centerX) / d;
            double deltaY = (h - centerY) / d;

            // distance or radius of dst image
            double dstR = sqrt(deltaX * deltaX + deltaY * deltaY);

            // distance or radius of src image (with formula)
            double srcR = (paramA * dstR * dstR * dstR + paramB * dstR * dstR + paramC * dstR + paramD) * dstR;

            // comparing old and new distance to get factor
            double factor = abs(dstR / srcR);

            // coordinates in source image
            double srcXd = centerX + (deltaX * factor * d);
            double srcYd = centerY + (deltaY * factor * d);

            // no interpolation yet (just nearest point)
            int srcX = (int) srcXd;
            int srcY = (int) srcYd;

			if (srcX >= 0 && srcY >= 0 && srcX < CB256->rows && srcY < CB256->rows) 
			{
                //int dstPos = h * CB256->rows + w;
				result.at<uchar>(h, w) = CB256->at<uchar>(srcY, srcX);
				//result.at<uchar>(srcY, srcX) = CB256->at<uchar>(h, w);
                //pixels[dstPos] = pixelsCopy[srcY * width + srcX];
            }
		}
	}

	imshow("123", result);
	cvWaitKey(0);		
}

Mat HW6_2_1()
{
int height,width,step,channels,depth;
uchar* data1;
CvMat *dft_A;
CvMat *dft_B;
CvMat *dft_C;
IplImage* im;
IplImage* im1;
IplImage* image_ReB;
IplImage* image_ImB;

IplImage* image_ReC;
IplImage* image_ImC;
IplImage* complex_ImC;
CvScalar val;
IplImage* k_image_hdr;
int i,j,k;

FILE *fp;
fp = fopen("test.txt","w+");
int dft_M,dft_N;
int dft_M1,dft_N1;

CvMat* cvShowDFT1(IplImage*, int, int,char*);
void cvShowInvDFT1(IplImage*, CvMat*, int, int,char*);

im1 = cvLoadImage("textbook_blur00065_688x688_1.png");
cvNamedWindow("Original-Color", 0);
cvShowImage("Original-Color", im1);
im = cvLoadImage("textbook_blur00065_688x688_1.png", CV_LOAD_IMAGE_GRAYSCALE );


cvNamedWindow("Original-Gray", 0);
cvShowImage("Original-Gray", im);
IplImage* k_image;
int rowLength= 11;
long double kernels[11*11];
CvMat kernel;
int x,y;
long double PI_F=3.14159265358979;

//long double SIGMA = 0.84089642;
long double SIGMA = 0.014089642;
//long double SIGMA = 0.00184089642;
long double EPS = 2.718;
long double numerator,denominator;
long double value,value1;
long double a,b,c,d;

numerator = (pow((float)-3,2) + pow((float) 0,2))/(2*pow((float)SIGMA,2));
printf("Numerator=%f\n",numerator);
denominator = sqrt((float) (2 * PI_F * pow(SIGMA,2)));
printf("denominator=%1.8f\n",denominator);

value = (pow((float)EPS, (float)-numerator))/denominator;
printf("Value=%1.8f\n",value);
for(x = -5; x < 6; x++){
for (y = -5; y < 6; y++)
{
//numerator = (pow((float)x,2) + pow((float) y,2))/(2*pow((float)SIGMA,2));
numerator = (pow((float)x,2) + pow((float)y,2))/(2.0*pow(SIGMA,2));
denominator = sqrt((2.0 * 3.14159265358979 * pow(SIGMA,2)));
value = (pow(EPS,-numerator))/denominator;
printf(" %1.8f ",value);
kernels[x*rowLength +y+55] = (float)value;

}
printf("\n");
}
printf("———————————\n");
for (i=-5; i < 6; i++){
for(j=-5;j < 6;j++){
printf(" %1.8f ",kernels[i*rowLength +j+55]);
}
printf("\n");
}
kernel= cvMat(rowLength, // number of rows
rowLength, // number of columns
CV_32FC1, // matrix data type
&kernels);
k_image_hdr = cvCreateImageHeader( cvSize(rowLength,rowLength), IPL_DEPTH_32F,1);
k_image = cvGetImage(&kernel,k_image_hdr);

height = k_image->height;
width = k_image->width;
step = k_image->widthStep/sizeof(float);
depth = k_image->depth;
channels = k_image->nChannels;
//data1 = (float *)(k_image->imageData);
data1 = (uchar *)(k_image->imageData);
cvNamedWindow("blur kernel", 0);
cvShowImage("blur kernel", k_image);

dft_M = cvGetOptimalDFTSize( im->height - 1 );
dft_N = cvGetOptimalDFTSize( im->width - 1 );
//dft_M1 = cvGetOptimalDFTSize( im->height+99 – 1 );
//dft_N1 = cvGetOptimalDFTSize( im->width+99 – 1 );
dft_M1 = cvGetOptimalDFTSize( im->height+3 - 1 );
dft_N1 = cvGetOptimalDFTSize( im->width+3 - 1 );
printf("dft_N1=%d,dft_M1=%d\n",dft_N1,dft_M1);

// Perform DFT of original image
dft_A = cvShowDFT1(im, dft_M1, dft_N1,"original");
//Perform inverse (check)
//cvShowInvDFT1(im,dft_A,dft_M1,dft_N1, "original"); – Commented as it overwrites the DFT
// Perform DFT of kernel
dft_B = cvShowDFT1(k_image,dft_M1,dft_N1,"kernel");
//Perform inverse of kernel (check)
//cvShowInvDFT1(k_image,dft_B,dft_M1,dft_N1, "kernel");//- Commented as it overwrites the DFT
// Multiply numerator with complex conjugate
dft_C = cvCreateMat( dft_M1, dft_N1, CV_64FC2 );
printf("%d %d %d %d\n",dft_M,dft_N,dft_M1,dft_N1);

// Multiply DFT(blurred image) * complex conjugate of blur kernel
cvMulSpectrums(dft_A,dft_B,dft_C,CV_DXT_MUL_CONJ);
//cvShowInvDFT1(im,dft_C,dft_M1,dft_N1,"blur1?);

// Split Fourier in real and imaginary parts
image_ReC = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 1);
image_ImC = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 1);
complex_ImC = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 2);
printf("%d %d %d %d\n", dft_M,dft_N,dft_M1,dft_N1);
//cvSplit( dft_C, image_ReC, image_ImC, 0, 0 );
cvSplit( dft_C, image_ReC, image_ImC, 0, 0 );

// Compute A^2 + B^2 of denominator or blur kernel
image_ReB = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 1);
image_ImB = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 1);

// Split Real and imaginary parts
cvSplit( dft_B, image_ReB, image_ImB, 0, 0 );
cvPow( image_ReB, image_ReB, 2.0);
cvPow( image_ImB, image_ImB, 2.0);
cvAdd(image_ReB, image_ImB, image_ReB,0);
val = cvScalarAll(kappa);
cvAddS(image_ReB,val,image_ReB,0);
//Divide Numerator/A^2 + B^2
cvDiv(image_ReC, image_ReB, image_ReC, 1.0);
cvDiv(image_ImC, image_ReB, image_ImC, 1.0);

// Merge Real and complex parts
cvMerge(image_ReC, image_ImC, NULL, NULL, complex_ImC);
// Perform Inverse
cvShowInvDFT1(im, (CvMat *)complex_ImC,dft_M1,dft_N1,"Weiner o/p k=10000 SIGMA=0.014089642");
cvSaveImage("HW6_2_1.png",im);
cvWaitKey(0);

return im;
}

CvMat* cvShowDFT1(IplImage* im, int dft_M, int dft_N,char* src)
{
IplImage* realInput;
IplImage* imaginaryInput;
IplImage* complexInput;
CvMat* dft_A, tmp;
IplImage* image_Re;
IplImage* image_Im;
char str[80];
double m, M;
realInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
imaginaryInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
complexInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 2);
cvScale(im, realInput, 1.0, 0.0);
cvZero(imaginaryInput);
cvMerge(realInput, imaginaryInput, NULL, NULL, complexInput);

dft_A = cvCreateMat( dft_M, dft_N, CV_64FC2 );
image_Re = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);
image_Im = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);

// copy A to dft_A and pad dft_A with zeros
cvGetSubRect( dft_A, &tmp, cvRect(0,0, im->width, im->height));
cvCopy( complexInput, &tmp, NULL );
if( dft_A->cols > im->width )
{
cvGetSubRect( dft_A, &tmp, cvRect(im->width,0, dft_A->cols - im->width, im->height));
cvZero( &tmp );
}
// no need to pad bottom part of dft_A with zeros because of
// use nonzero_rows parameter in cvDFT() call below

cvDFT( dft_A, dft_A, CV_DXT_FORWARD, complexInput->height );
strcpy(str,"DFT -");
strcat(str,src);
cvNamedWindow(str, 0);

// Split Fourier in real and imaginary parts
cvSplit( dft_A, image_Re, image_Im, 0, 0 );
// Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
cvPow( image_Re, image_Re, 2.0);
cvPow( image_Im, image_Im, 2.0);
cvAdd( image_Re, image_Im, image_Re, NULL);
cvPow( image_Re, image_Re, 0.5 );

// Compute log(1 + Mag)
cvAddS( image_Re, cvScalarAll(1.0), image_Re, NULL ); // 1 + Mag
cvLog( image_Re, image_Re ); // log(1 + Mag)
cvMinMaxLoc(image_Re, &m, &M, NULL, NULL, NULL);
cvScale(image_Re, image_Re, 1.0/(M-m), 1.0*(-m)/(M-m));
cvShowImage(str, image_Re);
return(dft_A);
}

void cvShowInvDFT1(IplImage* im, CvMat* dft_A, int dft_M, int dft_N,char* src)
{
IplImage* realInput;
IplImage* imaginaryInput;
IplImage* complexInput;
IplImage * image_Re;
IplImage * image_Im;
double m, M;
char str[80];
realInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
imaginaryInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 1);
complexInput = cvCreateImage( cvGetSize(im), IPL_DEPTH_64F, 2);
image_Re = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);
image_Im = cvCreateImage( cvSize(dft_N, dft_M), IPL_DEPTH_64F, 1);

//cvDFT( dft_A, dft_A, CV_DXT_INV_SCALE, complexInput->height );
cvDFT( dft_A, dft_A, CV_DXT_INV_SCALE, dft_M);
strcpy(str,"DFT INVERSE – ");
strcat(str,src);
cvNamedWindow(str, 0);
// Split Fourier in real and imaginary parts
cvSplit( dft_A, image_Re, image_Im, 0, 0 );
// Compute the magnitude of the spectrum Mag = sqrt(Re^2 + Im^2)
cvPow( image_Re, image_Re, 2.0);
cvPow( image_Im, image_Im, 2.0);
cvAdd( image_Re, image_Im, image_Re, NULL);
cvPow( image_Re, image_Re, 0.5 );

// Compute log(1 + Mag)
cvAddS( image_Re, cvScalarAll(1.0), image_Re, NULL ); // 1 + Mag
cvLog( image_Re, image_Re ); // log(1 + Mag)
cvMinMaxLoc(image_Re, &m, &M, NULL, NULL, NULL);
cvScale(image_Re, image_Re, 1.0/(M-m), 1.0*(-m)/(M-m));
//cvCvtColor(image_Re, image_Re, CV_GRAY2RGBA);
cvShowImage(str, image_Re);

}


Mat HW6_2_2()
{
int height,width,step,channels,depth;
uchar* data1;
CvMat *dft_A;
CvMat *dft_B;
CvMat *dft_C;
IplImage* im;
IplImage* im1;
IplImage* image_ReB;
IplImage* image_ImB;

IplImage* image_ReC;
IplImage* image_ImC;
IplImage* complex_ImC;
CvScalar val;
IplImage* k_image_hdr;
int i,j,k;

FILE *fp;
fp = fopen("test.txt","w+");
int dft_M,dft_N;
int dft_M1,dft_N1;

CvMat* cvShowDFT1(IplImage*, int, int,char*);
void cvShowInvDFT1(IplImage*, CvMat*, int, int,char*);

im1 = cvLoadImage("textbook_blur65_688x688_1.png");
cvNamedWindow("Original-Color", 0);
cvShowImage("Original-Color", im1);
im = cvLoadImage("textbook_blur65_688x688_1.png", CV_LOAD_IMAGE_GRAYSCALE );


cvNamedWindow("Original-Gray", 0);
cvShowImage("Original-Gray", im);
IplImage* k_image;
int rowLength= 11;
long double kernels[11*11];
CvMat kernel;
int x,y;
long double PI_F=3.14159265358979;

//long double SIGMA = 0.84089642;
long double SIGMA = 0.014089642;
//long double SIGMA = 0.00184089642;
long double EPS = 2.718;
long double numerator,denominator;
long double value,value1;
long double a,b,c,d;

numerator = (pow((float)-3,2) + pow((float) 0,2))/(2*pow((float)SIGMA,2));
printf("Numerator=%f\n",numerator);
denominator = sqrt((float) (2 * PI_F * pow(SIGMA,2)));
printf("denominator=%1.8f\n",denominator);

value = (pow((float)EPS, (float)-numerator))/denominator;
printf("Value=%1.8f\n",value);
for(x = -5; x < 6; x++){
for (y = -5; y < 6; y++)
{
//numerator = (pow((float)x,2) + pow((float) y,2))/(2*pow((float)SIGMA,2));
numerator = (pow((float)x,2) + pow((float)y,2))/(2.0*pow(SIGMA,2));
denominator = sqrt((2.0 * 3.14159265358979 * pow(SIGMA,2)));
value = (pow(EPS,-numerator))/denominator;
printf(" %1.8f ",value);
kernels[x*rowLength +y+55] = (float)value;

}
printf("\n");
}
printf("———————————\n");
for (i=-5; i < 6; i++){
for(j=-5;j < 6;j++){
printf(" %1.8f ",kernels[i*rowLength +j+55]);
}
printf("\n");
}
kernel= cvMat(rowLength, // number of rows
rowLength, // number of columns
CV_32FC1, // matrix data type
&kernels);
k_image_hdr = cvCreateImageHeader( cvSize(rowLength,rowLength), IPL_DEPTH_32F,1);
k_image = cvGetImage(&kernel,k_image_hdr);

height = k_image->height;
width = k_image->width;
step = k_image->widthStep/sizeof(float);
depth = k_image->depth;
channels = k_image->nChannels;
//data1 = (float *)(k_image->imageData);
data1 = (uchar *)(k_image->imageData);
cvNamedWindow("blur kernel", 0);
cvShowImage("blur kernel", k_image);

dft_M = cvGetOptimalDFTSize( im->height - 1 );
dft_N = cvGetOptimalDFTSize( im->width - 1 );
//dft_M1 = cvGetOptimalDFTSize( im->height+99 – 1 );
//dft_N1 = cvGetOptimalDFTSize( im->width+99 – 1 );
dft_M1 = cvGetOptimalDFTSize( im->height+3 - 1 );
dft_N1 = cvGetOptimalDFTSize( im->width+3 - 1 );
printf("dft_N1=%d,dft_M1=%d\n",dft_N1,dft_M1);

// Perform DFT of original image
dft_A = cvShowDFT1(im, dft_M1, dft_N1,"original");
//Perform inverse (check)
//cvShowInvDFT1(im,dft_A,dft_M1,dft_N1, "original"); – Commented as it overwrites the DFT
// Perform DFT of kernel
dft_B = cvShowDFT1(k_image,dft_M1,dft_N1,"kernel");
//Perform inverse of kernel (check)
//cvShowInvDFT1(k_image,dft_B,dft_M1,dft_N1, "kernel");//- Commented as it overwrites the DFT
// Multiply numerator with complex conjugate
dft_C = cvCreateMat( dft_M1, dft_N1, CV_64FC2 );
printf("%d %d %d %d\n",dft_M,dft_N,dft_M1,dft_N1);

// Multiply DFT(blurred image) * complex conjugate of blur kernel
cvMulSpectrums(dft_A,dft_B,dft_C,CV_DXT_MUL_CONJ);
//cvShowInvDFT1(im,dft_C,dft_M1,dft_N1,"blur1?);

// Split Fourier in real and imaginary parts
image_ReC = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 1);
image_ImC = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 1);
complex_ImC = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 2);
printf("%d %d %d %d\n", dft_M,dft_N,dft_M1,dft_N1);
//cvSplit( dft_C, image_ReC, image_ImC, 0, 0 );
cvSplit( dft_C, image_ReC, image_ImC, 0, 0 );

// Compute A^2 + B^2 of denominator or blur kernel
image_ReB = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 1);
image_ImB = cvCreateImage( cvSize(dft_N1, dft_M1), IPL_DEPTH_64F, 1);

// Split Real and imaginary parts
cvSplit( dft_B, image_ReB, image_ImB, 0, 0 );
cvPow( image_ReB, image_ReB, 2.0);
cvPow( image_ImB, image_ImB, 2.0);
cvAdd(image_ReB, image_ImB, image_ReB,0);
val = cvScalarAll(kappa);
cvAddS(image_ReB,val,image_ReB,0);
//Divide Numerator/A^2 + B^2
cvDiv(image_ReC, image_ReB, image_ReC, 1.0);
cvDiv(image_ImC, image_ReB, image_ImC, 1.0);

// Merge Real and complex parts
cvMerge(image_ReC, image_ImC, NULL, NULL, complex_ImC);
// Perform Inverse
cvShowInvDFT1(im, (CvMat *)complex_ImC,dft_M1,dft_N1,"Weiner o/p k=10000 SIGMA=0.014089642");
cvSaveImage("HW6_2_2.png",im);
cvWaitKey(0);

return im;
}













int main(int argc, char** argv)
{
	Q1();
	Q1_2();
	 HW6_2_1();
		 HW6_2_2();
}