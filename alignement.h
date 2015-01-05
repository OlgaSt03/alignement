//

#ifndef XXX2_H
#define XXX2_H
#include <cv.h>

void init_warp(cv::Mat W, float wz, float tx, float ty);
void warp_image(cv::Mat pSrcFrame, cv::Mat pDstFrame, cv::Mat W);
void draw_warped_rect(cv::Mat pImage, cv::Rect rect, cv::Mat W);


template <class T>
float interpolate2(cv::Mat pImage, float x, float y)
{
	// Get the nearest integer pixel coords (xi;yi).
	int xi = cvFloor(x);
	int yi = cvFloor(y);

	float k1 = x-xi; // Coefficients for interpolation formula.
	float k2 = y-yi;

	int f1 = xi<pImage.rows-1;  // Check that pixels to the right  
	int f2 = yi<pImage.cols-1; // and to down direction exist.


	T *row1 = &pImage.at<T>(yi,xi);
	T *row2 = &pImage.at<T>(yi,xi+1);
				
	// Interpolate pixel intensity.
	float interpolated_value = (1.0f-k1)*(1.0f-k2)*(float)row1[0] +
				(f1 ? ( k1*(1.0f-k2)*(float)row1[1] ):0) +
				(f2 ? ( (1.0f-k1)*k2*(float)row2[0] ):0) +						
				((f1 && f2) ? ( k1*k2*(float)row2[1] ):0) ;


	return interpolated_value;
}

#define SET_VECTOR(X, u, v)\
	X.at<float>(0,0) = (float)(u);\
	X.at<float>(1,0) = (float)(v);\
	X.at<float>(2,0) = 1.0f;

	//CV_MAT_ELEM((X), float, 0, 0) = (float)(u);\
	//CV_MAT_ELEM((X), float, 1, 0) = (float)(v);\
	//CV_MAT_ELEM((X), float, 2, 0) = 1.0f;

#define GET_VECTOR(X, u, v)\
	u = X.at<float>(0,0);\
	v = X.at<float>(1,0);

	//(u) = CV_MAT_ELEM((X), float, 0, 0);\
	//(v) = CV_MAT_ELEM((X), float, 1, 0);

#define GET_INT_VECTOR(X, u, v)\
	u = (int)X.at<float>(0,0);\
	v = (int)X.at<float>(1,0);
	//(u) = (int)CV_MAT_ELEM((X), float, 0, 0);\
	//(v) = (int)CV_MAT_ELEM((X), float, 1, 0);

	
#endif	/* XXX2_H */

////////////////////////////////////////////////////////////////////////
///////////////                                  //////////////////////
////////////////////////////////////////////////////////////////////////
#include <xxx2.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

// Our warp matrix looks like this one:
//
//  ! 1  -wz  tx !
//  ! wz  1   ty !
//  ! 0   0   1  !
//

void init_warp(cv::Mat W, float wz, float tx, float ty)
{
	W.at<float>(0,0) = 1;
	W.at<float>(1,0) = wz;
	W.at<float>(2,0) = 0;

	W.at<float>(0,1) = -wz;
	W.at<float>(1,1) = 1;
	W.at<float>(2,2) = 0;
	
	W.at<float>(0,2) = tx;
	W.at<float>(1,2) = ty;
	W.at<float>(2,2) = 1;

}

void warp_image(cv::Mat pSrcFrame, cv::Mat pDstFrame, cv::Mat W)
{
	pDstFrame=0;

	cv::Mat X = cv::Mat::zeros(3,1, CV_32F);
	cv::Mat Z = cv::Mat::zeros(3,1, CV_32F);
	int x, y;

	for(x=0;x<pSrcFrame.rows; x++)
	{
		for(y=0;y<pSrcFrame.cols; y++)
		{
			SET_VECTOR(X, x, y);

			gemm(W, X, 1.0, 0, 0, Z);

			int x2, y2;
			GET_INT_VECTOR(Z, x2, y2);

			if(x2>=0 && x2<pDstFrame.rows &&
				y2>=0 && y2<pDstFrame.cols)
			{
				//CV_IMAGE_ELEM(pDstFrame, uchar, y2, x2) = CV_IMAGE_ELEM(pSrcFrame,uchar, y, x);
				pDstFrame.at<float>(y2,x2)=pDstFrame.at<float>(y,x);
			}
		}
	}

	//cvSmooth(pDstFrame, pDstFrame);
	blur( pDstFrame, pDstFrame, Size( 3, 3 ) );

	X.release();
	Z.release();

}
void draw_warped_rect(cv::Mat pImage, cv::Rect rect, cv::Mat W)
{
	cv::Point lt, lb, rt, rb;
	
	
	cv::Mat X = cv::Mat::zeros(3,1, CV_32F);
	cv::Mat Z = cv::Mat::zeros(3,1, CV_32F);
	int x, y;

	// left-top point
	SET_VECTOR(X, rect.x, rect.y);
	gemm(W, X, 1, 0, 0, Z);
	GET_INT_VECTOR(Z, lt.x, lt.y);

	// left-bottom point
	SET_VECTOR(X, rect.x, rect.y+rect.height);
	gemm(W, X, 1, 0, 0, Z);
	GET_INT_VECTOR(Z, lb.x, lb.y);

	// right-top point
	SET_VECTOR(X, rect.x+rect.width, rect.y);
	gemm(W, X, 1, 0, 0, Z);
	GET_INT_VECTOR(Z, rt.x, rt.y);

	// right-bottom point
	SET_VECTOR(X, rect.x+rect.width, rect.y+rect.height);
	gemm(W, X, 1, 0, 0, Z);
	GET_INT_VECTOR(Z, rb.x, rb.y);

	// draw rectangle
	//cvLine(&pImage, lt, rt, cvScalar(255));
	//cvLine(&pImage, rt, rb, cvScalar(255));
	//cvLine(&pImage, rb, lb, cvScalar(255));
	//cvLine(&pImage, lb, lt, cvScalar(255));
	// draw rectangle
	line(pImage, lt, rt, 255, 1, 8);
	line(pImage, rt, rb, 255, 1, 8);
	line(pImage, rb, lb, 255, 1, 8);
	line(pImage, lb, lt, 255, 1, 8);

	// release resources and exit
	X.release();
	Z.release();
	
}


///////////////////////////////////////////////////////////////////////////////
///                                                                  ///////////
/////////////////////////////////////////////////////////////////////////////////

#ifndef ALIGNEMENT_H
#define ALIGNEMENT_H

#include "cv.h"

void align_image_inverse_compositional(cv::Mat pImgT, cv::Rect omega, cv::Mat pImgI);

#endif //__ALIGNEMENT_H

////////////////////////////////////////////////////////////////////////
///////////////                                  //////////////////////
////////////////////////////////////////////////////////////////////////
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>//Sobel 
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <stdlib.h>
#include "stdio.h"
#include <xxx2.h>
#include <time.h>



using namespace cv;
using namespace std;

void align_image_inverse_compositional(cv::Mat pImgT, cv::Rect omega, cv::Mat pImgI)
{
	

	// Some constants for iterative minimization process.
	const float EPS = 1E-5f; // Threshold value for termination criteria.
	const int MAX_ITER = 100;  // Maximum iteration count.

	Mat W = Mat::zeros(3,3, CV_32F);
	//cout << "W = " << endl << " " << W << endl << endl;

	W.at<float>(0,0) =1;
	//cout << "W 2= " << endl << " " << W << endl << endl;

	Mat dW = Mat::zeros(3,3, CV_32F);
	//cout << "dW = " << endl << " " << dW << endl << endl;
 
	Mat idW = Mat::zeros(3,3, CV_32F);
	//cout << "idW = " << endl << " " << idW << endl << endl;

	Mat X = Mat::zeros(3,1, CV_32F);
	//cout << "X = " << endl << " " << X << endl << endl;

	Mat Z = Mat::zeros(3,1, CV_32F);
	cout << "Z = " << endl << " " << Z << endl << endl;

	cv::Mat H = cv::Mat::zeros(3,3, CV_32F);
	//cout << "H = " << endl << " " << H << endl << endl;

	Mat iH = Mat::zeros(3,3, CV_32F);
	//cout << "iH = " << endl << " " << iH << endl << endl;

	Mat b = Mat::zeros(3,1, CV_32F);
	//cout << "b = " << endl << " " << b << endl << endl;

	Mat delta_p = Mat::zeros(3,1, CV_32F);
	//cout << "delta_p = " << endl << " " << delta_p << endl << endl;
 
	Size image_size = pImgI.size();//taille d'image
	cout << "Taille d'image = " << endl << " " << image_size << endl << endl;

	//Calcul des gradients
	Mat pGradTx;
	Mat pGradTy;
	Mat pStDesc;

	pGradTx.create( pImgI.size(), CV_16S );
	imshow("pGradTx",pGradTx);
	
	pGradTy.create( pImgI.size(), CV_16S );
	imshow("pGradTy",pGradTy);

	pStDesc.create( pImgI.size(), CV_32F );
	imshow("pStDesc",pStDesc);


	// Get current time. We will use it later to obtain total calculation time.
	clock_t start_time = clock();

	Mat abs_pGradTx, abs_pGradTy;
	//Calcule dradient de T
	cv::Sobel(pImgT, pGradTx,pImgT.depth(), 1,0);//Gradient dans la direction X
	convertScaleAbs( pGradTx, abs_pGradTx ,0.125);//Normalisation
	cv::Sobel(pImgT, pGradTy,pImgT.depth(), 0,1);//Gradient dans la direction Y
	convertScaleAbs( pGradTy, abs_pGradTy, 0.125 );//Normalisation

	//Descente de Gradient et matrice Hessienne
	//H=0;//pas sur---cvSet(H, cvScalar(0)); // Set Hessian with zeroes


	int u, v;	// (u,v) - pixel coordinates in the coordinate frame of T.
	//int pImage
	float u2, v2; // (u2,v2) - pixel coordinates in the coordinate frame of I.
	//cv::Rect omega = cv::Rect(110, 100, 200, 150);

	// Walk through pixels in the template T.
	int i, j;
	for(i=0; i<omega.width; i++)
	{
		u = i + omega.x;

		for(j=0; j<omega.height; j++)
		{
			v = j + omega.y;

			// Evaluate gradient of T.
			float Tx = pGradTx.at<float>(v,u);	
			float Ty = pGradTy.at<float>(v,u);
			
			// Calculate steepest descent image's element.
			float *stdesc(0);
			stdesc= &pStDesc.at<float>(v,u*3); // an element of steepest descent image
			stdesc[0] = (float)(-v*Tx+u*Ty);
			stdesc[1] = (float)Tx;
			stdesc[2] = (float)Ty;

			// Add a term to Hessian.
			int l,m;
			for(l=0;l<3;l++)
			{
				for(m=0;m<3;m++)
				{
					H.at<float>(l,m) += stdesc[l]*stdesc[m];
				}
			}
		}	
	}

// Invert Hessian.
	double inv_res = cv::invert(H, iH);
	if(inv_res==0)
	{
		cout << "Error: Hessian is singular " << endl;//printf("Error: Hessian is singular.\n");
		return;
	}

	//Iteration stage.


	// Set warp with identity.
	//cvSetIdentity(W);
	W = Mat::eye(3, 3, CV_32F);

	// Here we will store current value of mean error.
	float mean_error=0;

	// Iterate
	int iter=0; // number of current iteration
	//int MAX_ITER=100;
	while(iter < MAX_ITER)
	{
		iter++; // Increment iteration counter

		mean_error = 0; // Set mean error value with zero

		int pixel_count=0; // Count of processed pixels
		
		//cvSet(b, cvScalar(0)); // Set b matrix with zeroes
			
		// Walk through pixels in the template T.
		int i, j;
		for(i=0; i<omega.width; i++)
		{
			int u = i + omega.x;

			for(j=0; j<omega.height; j++)
			{
				int v = j + omega.y;

				// Set vector X with pixel coordinates (u,v,1)
				SET_VECTOR(X, u, v);

				// Warp Z=W*X
				gemm(W, X, 1, 0, 0, Z);

				// Get coordinates of warped pixel in coordinate frame of I.
				GET_VECTOR(Z, u2, v2);

				// Get the nearest integer pixel coords (u2i;v2i).
				int u2i = cvFloor(u2);
				int v2i = cvFloor(v2);

				if(u2i>=0 && u2i<pImgI.rows && // check if pixel is inside I.
					v2i>=0 && v2i<pImgI.cols)
				{
					pixel_count++;

					// Calculate intensity of a transformed pixel with sub-pixel accuracy
					// using bilinear interpolation.
					float I2 = interpolate2<float>(pImgI, u2, v2);

					// Calculate image difference D = I(W(x,p))-T(x).
					float D = I2 - pImgT.at<float>(v,u);

					// Update mean error value.
					mean_error += fabs(D);

					// Add a term to b matrix.
					float* stdesc(0);
						stdesc= &pStDesc.at<float>(v,u*3);
					float* pb(0);
						pb= &b.at<float>(0,0);
					pb[0] += stdesc[0] * D;
					pb[1] += stdesc[1] * D;
					pb[2] += stdesc[2] * D;					
				}	
			}
		}

		// Finally, calculate resulting mean error.
		if(pixel_count!=0)
			mean_error /= pixel_count;

		// Find parameter increment. 
		gemm(iH, b, 1, 0, 0, delta_p);
		float delta_wz = delta_p.at<float>(0,0);
		float delta_tx = delta_p.at<float>(1,0);
		float delta_ty = delta_p.at<float>(2,0);

		init_warp(dW, delta_wz, delta_tx, delta_ty);
		// Invert warp.
		inv_res = cv::invert(dW, idW);
		if(inv_res==0)
		{
			cout << "Error: Warp is singular " << endl;//printf("Error: Warp matrix is singular.\n");
			return;
		}

		gemm(idW, W, 1, 0, 0, dW);
		//cvCopy(dW, W);
		//W = dW.clone();
		dW.copyTo(W);

		// Print diagnostic information to screen.
		printf("iter=%d dwz=%f dtx=%f dty=%f mean_error=%f\n", iter, delta_wz, delta_tx, delta_ty, mean_error);

		// Check termination critera.
		if(fabs(delta_wz)<=EPS && fabs(delta_tx)<=EPS && fabs(delta_ty)<=EPS) break;
	}

	// Get current time and obtain total time of calculation.
	clock_t finish_time = clock();
	double total_time = (double)(finish_time-start_time)/CLOCKS_PER_SEC;

	// Print summary.
	printf("===============================================\n");
	printf("Algorithm: inverse compositional.\n");
	printf("Caclulation time: %g sec.\n", total_time);
	printf("Iteration count: %d\n", iter);
	printf("Epsilon: %f\n", EPS);
	printf("Resulting mean error: %f\n", mean_error);
	printf("===============================================\n");

	//// Free used resources and exit.
	W.release();
	dW.release();
	idW.release();
	H.release();
	iH.release();
	b.release();
	delta_p.release();
	X.release();
	Z.release();
	//waitKey();	
    //return 0;
	//system("PAUSE");
}


