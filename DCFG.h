#pragma once

#ifndef _DCFG_
#define _DCFG_

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

//===========================================================
//	parameters
//===========================================================
#define BLOCK_SIZE						(4.0)
#define BLOCK_SIZE_SQR					(16.0)
#define VARIANCE_INTERPOLATE_PARAM	    (1.0)

#define MAX_BG_AGE						(30.0)
#define VAR_MIN_NOISE_T			        (50.0*50.0)
#define VAR_DEC_RATIO			        (0.001)
#define MIN_BG_VAR						(5.0*5.0)	//15*15
#define INIT_BG_VAR						(20.0*20.0)	//15*15
#define INIT_FG_VAR						(20.0*20.0)	//15*15
#define MIN_FG_VAR						(5.0*5.0)	//15*15
#define MAX_FG_VAR						(50.0*50.0)	//15*15

#define NUM_MODELS						(2)
#define NUM_MODELS_FG					(5)
#define VAR_THRESH_BG_DETERMINE			(4.0) //-- theta_d in the paper section 2.4
#define VAR_THRESH_FG_DETERMINE			(8.0) //-- theta_d in the paper section 2.4
#define VAR_THRESH_MODEL_MATCH			(2.0)

#define GRID_SIZE_W						(32)
#define GRID_SIZE_H						(24)
#define WIN_SIZE						(10) //-- optical flow window size

#define Old_or_New						(0) // 0: old, 1: new


class DCFG
{

public:
	~DCFG();
	DCFG(const cv::Mat& frame);

	void apply(const cv::Mat& frame, cv::Mat& fgMask, cv::Mat& bg);

	//-- oldMask is the result of paper: Scene conditional background update for moving object detection in a moving camera
	cv::Mat debugImg, oldMask; 

private:
	//-----------------
	//-- global variables
	//-----------------
	cv::Size frameSize;
	int nChannels;
	cv::Mat frameGray, prevGray, fgMaskDilate, fgCandidate;
	float illum = 0.0; //-- illumination change
	float bgMean = 0.0;
	int frameNum = 0;

	//-----------------
	//-- Gaussian modeling
	//-----------------
	float* m_DistImg;
	float* m_DistImg_FG[NUM_MODELS_FG];

	float* m_Mean[NUM_MODELS];
	float* m_Var[NUM_MODELS];
	float* m_Age[NUM_MODELS];

	float* m_Mean_Temp[NUM_MODELS];
	float* m_Var_Temp[NUM_MODELS];
	float* m_Age_Temp[NUM_MODELS];

	float f_Mean[NUM_MODELS_FG];
	float f_Var[NUM_MODELS_FG];
	float f_Age[NUM_MODELS_FG];
	float f_Weight[NUM_MODELS_FG];
	int f_Num = 0;

	int* m_ModelIdx;
	int f_ModelIdx;

	int modelWidth, modelHeight, modelSize;

	int obsWidth, obsHeight, obsSize;

	void motionCompensate(double h[9]);
	void update(cv::Mat& pOutputImg);
	void updateFG(float input);

	//-----------------
	//-- KLT
	//-----------------
	cv::Mat eig, temp, maskimg;
	std::deque<cv::Mat> flowSeq, frameSeq;
	cv::Mat cumulativeFlow;

	// For LK
	std::vector<cv::Point2f> points[2];
	cv::TermCriteria termcrit;
	cv::Size winSize;
	std::vector<uchar> status;
	std::vector<float> err;
	cv::Point2f tempP;
	size_t count{};

	double matH[9]; //-- Homography Matrix
	void KLT(cv::Mat& imgGray);
	int nMatch[200];

	//-- Foreground motion estimation
	cv::Mat blockMag, blockAng;
	float fgVelocity = 0.0, fgAngle = 0.0;
	int fgPixelNum = 0;

	//-----------------
	//-- debug
	//-----------------
	cv::Mat testImg, colorFrame, prevColorFrame;
	std::fstream outFile;

	void draw_1d_gaussian(cv::Mat img, float mean, float stddev, cv::Scalar color);
	void draw_1d_draw_mixture(cv::Mat img, float* data, float* weights, int count_per_component, int num_components);
	void draw_1d_gaussian_mixture(cv::Mat img, float* mean, float* stddev, float* pi, int num_components);

	//-----------------
	//-- ViBe
	//-----------------
	//-- parameters

	// Background model
	const int numberSamples = 20;
	std::vector<cv::Mat>  backgroundModel;
	// Parameters for classification judgment of pixels
	const int minMatch = 2;
	int distanceThreshold = 20;
	// Background model update probability
	int updateFactor = 0;
	// 8-field (3 x 3)
	const int neighborWidth = 3;
	const int neighborHeight = 3;
	// Foreground and background segmentation
	const static  unsigned char BACK_GROUND;
	const static  unsigned char FORE_GROUND;
	// BGR distance calculation
	int distanceL1(const cv::Vec3b& src1, const  cv::Vec3b& src2);
	float  distanceL2(const cv::Vec3b& src1, const 	cv::Vec3b& src2);
	cv::Mat seg;

	// Gray image
	void originalVibe_Init_GRAY(const cv::Mat& firstFrame);
	void originalVibe_ClassifyAndUpdate_GRAY(const cv::Mat& frame, cv::OutputArray& _segmentation);
	// RGB three channels
	void originalVibe_Init_BGR(const cv::Mat& firstFrame);
	void originalVibe_ClassifyAndUpdate_BGR(const cv::Mat& frame, cv::OutputArray& _segmentation);

};

#endif _DCFG_