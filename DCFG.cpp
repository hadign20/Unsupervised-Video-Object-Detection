#include "DCFG.h"
#include <Windows.h>

#define DFCG_DEBUG 1

std::vector<std::vector<cv::Point>> findBiggestContour(cv::Mat& img, int numOfContours);

//-----------------
//-- timing debug
//-----------------
timeval tic, toc, tic_total, toc_total;
float rt_preProc;	// pre Processing time
float rt_motionComp;	// motion Compensation time
float rt_modelUpdate;	// model update time
float rt_total;		// Background Subtraction time

#if defined _WIN32 || defined _WIN64
int gettimeofday(struct timeval* tp, int* tz)
{
	LARGE_INTEGER tickNow;
	static LARGE_INTEGER tickFrequency;
	static BOOL tickFrequencySet = FALSE;
	if (tickFrequencySet == FALSE) {
		QueryPerformanceFrequency(&tickFrequency);
		tickFrequencySet = TRUE;
	}
	QueryPerformanceCounter(&tickNow);
	tp->tv_sec = (long)(tickNow.QuadPart / tickFrequency.QuadPart);
	tp->tv_usec = (long)(((tickNow.QuadPart % tickFrequency.QuadPart) * 1000000L) / tickFrequency.QuadPart);

	return 0;
}
#else
#include <sys/time.h>
#endif
//-----------------

DCFG::DCFG(const cv::Mat& frame) : frameSize(frame.size()) {
	//------------------------
	//-- Global variables
	//------------------------
	cv::cvtColor(frame, frameGray, CV_RGB2GRAY);
	//cv::medianBlur(frameGray, frameGray, 5);
	frameGray.copyTo(prevGray);
	frame.copyTo(colorFrame);
	frame.copyTo(prevColorFrame);
	oldMask = cv::Mat(frameSize, CV_8UC1,0.0);
	fgMaskDilate = cv::Mat(frameSize, CV_8UC1,0.0);
	fgCandidate = cv::Mat(frameSize, CV_8UC1,0.0);
	blockMag = cv::Mat(frameSize, CV_32FC1,0.0);
	blockAng = cv::Mat(frameSize, CV_32FC1,0.0);
	

	//------------------------
	//-- Gaussian model
	//------------------------
	m_DistImg = 0;
	
	for (int i = 0; i < NUM_MODELS; ++i) {
		m_Mean[i] = 0;
		m_Var[i] = 0;
		m_Age[i] = 0;
		m_Mean_Temp[i] = 0;
		m_Var_Temp[i] = 0;
		m_Age_Temp[i] = 0;
	}

	for (int i = 0; i < NUM_MODELS_FG; ++i) {
		f_Mean[i] = 0;
		f_Var[i] = 0;
		f_Age[i] = 0;
		m_DistImg_FG[i] = 0;
	}

	obsWidth = frame.cols; //-- observation (frame) width
	obsHeight = frame.rows;
	obsSize = obsWidth * obsHeight;

	modelWidth = obsWidth / BLOCK_SIZE; //-- bg model width (which is smaller because we use blocks instead of pixels)
	modelHeight = obsHeight / BLOCK_SIZE;
	modelSize = modelWidth * modelHeight; //-- total number of pixels in the bg model (bg image)

	//-- Initialize Storage
	m_DistImg = new float[obsSize] {};
	m_ModelIdx = new int[modelSize] {};

	for (int i = 0; i < NUM_MODELS_FG; ++i)
		m_DistImg_FG[i] = new float[obsSize] {};

	for (int i = 0; i < NUM_MODELS; ++i) {
		m_Mean[i] = new float[modelSize] {};
		m_Var[i] = new float[modelSize] {};
		m_Age[i] = new float[modelSize] {};

		m_Mean_Temp[i] = new float[modelSize] {};
		m_Var_Temp[i] = new float[modelSize] {};
		m_Age_Temp[i] = new float[modelSize] {};
	}

	//------------------------
	//-- KLT
	//------------------------
	termcrit = cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
	winSize = cv::Size(WIN_SIZE, WIN_SIZE);
	//-- Init homography
	for (int i = 0; i < 9; i++)
		matH[i] = i / 3 == i % 3 ? 1 : 0;

	//------------------------
	//-- ViBe
	//------------------------
	//originalVibe_Init_BGR(frame);
	originalVibe_Init_GRAY(frameGray);

	//outFile.open("D:/Project/shadow/results/log.txt");
}

DCFG::~DCFG() {
	delete[] m_Mean;
	delete[] m_Var;
	delete[] m_Age;
	delete[] m_Mean_Temp;
	delete[] m_Var_Temp;
	delete[] m_Age_Temp;
	delete[] m_DistImg;
	delete[] m_DistImg_FG;
	delete[] m_ModelIdx;
	outFile.close();
}


//------------------------
//-- MCD
//------------------------
void DCFG::apply(const cv::Mat& frame, cv::Mat& fgMask, cv::Mat& bg)
{
	frameNum++;
	fgCandidate.setTo(0);

#if DFCG_DEBUG
	frame.copyTo(colorFrame);
	cv::cvtColor(frame, frameGray, CV_RGB2GRAY);
	cv::medianBlur(frameGray, frameGray, 5);
	if (!fgMask.empty()) cv::dilate(fgMask, fgMaskDilate, cv::Mat(), cv::Point(-1, -1), 8, 1, 1);

	gettimeofday(&tic, NULL);
#endif

	//-- pre processing
	


#if DFCG_DEBUG
	gettimeofday(&toc, NULL);
	rt_preProc = (float)(toc.tv_usec - tic.tv_usec) / 1000.0;
	gettimeofday(&tic, NULL);
#endif

	//-- optical flow
	KLT(frameGray);
	//-- motion compensation
	motionCompensate(matH);

#if DFCG_DEBUG
	gettimeofday(&toc, NULL);
	rt_motionComp = (float)(toc.tv_usec - tic.tv_usec) / 1000.0;

	//-- background model
	gettimeofday(&tic, NULL);
#endif
	//-- Update BG Model and Detect
	update(fgMask);

#if DFCG_DEBUG
	gettimeofday(&toc, NULL);
	rt_modelUpdate = (float)(toc.tv_usec - tic.tv_usec) / 1000.0;
	rt_total = rt_preProc + rt_motionComp + rt_modelUpdate;
#endif
	
	
	//-- Illumination change estimation (paper: Scene conditional background update for moving object detection in a moving camera)
	cv::Scalar frameMean = cv::mean(frameGray, 255 - fgMask);
	bgMean /= modelSize;
	illum = frameMean[0] - bgMean;
	//std::cout << "frame: " << frameMean[0] << " bg: " << bgMean << std::endl;


#if DFCG_DEBUG
	// Debug display of individual maps
	//cv::Mat mean = cv::Mat(modelHeight, modelWidth, CV_32F, m_Mean[0]);
	//cv::resize(mean, mean, cv::Size(), BLOCK_SIZE, BLOCK_SIZE);
	//cv::imshow("mean", mean / 255.0);
	//cv::Mat var = cv::Mat(modelHeight, modelWidth, CV_32F, m_Var[0]);
	//cv::resize(var, var, cv::Size(), BLOCK_SIZE, BLOCK_SIZE);
	//cv::imshow("var", var / 2550.0);
	//cv::Mat age = cv::Mat(modelHeight, modelWidth, CV_32F, m_Age[0]);
	//cv::resize(age, age, cv::Size(), BLOCK_SIZE, BLOCK_SIZE);
	//cv::imshow("age", age / 50.0);


	//-- draw normal chart
	//const int graph_width = 640;
	//const int graph_height = 140;
	//cv::Mat graph(graph_height, graph_width, CV_8UC3, cv::Scalar(255, 255, 255));
	//cv::Scalar color = cv::Scalar(0, 0, 0);
	//draw_1d_gaussian_mixture(graph, f_Mean, f_Var, f_Weight, f_Num);
	//cv::imshow("graph", graph);

	//////////////////////////////////////////////////////////////////////////
	// Debug Output
	for (int i = 0; i < 100; ++i) {
		printf("\b");
	}
	printf("PP: %.2f(ms)\tOF: %.2f(ms)\tBGM: %.2f(ms)\tTotal time: \t%.2f(ms)", MAX(0.0, rt_preProc), MAX(0.0, rt_motionComp), MAX(0.0, rt_modelUpdate), MAX(0.0, rt_total));
	frame.copyTo(prevColorFrame);

	//-- red mask
	//cv::Mat redMask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
	//redMask.setTo(cv::Scalar(0, 0, 255), fgMask);
	//debugImg = frame + .5 * redMask;
	//cv::imshow("debugImg", debugImg);

	//-- fgMask & frame
	cv::Mat fgMaskBlur;
	cv::medianBlur(fgMask, fgMaskBlur, 5);
	cv::hconcat(frameGray, fgMaskBlur, debugImg);
	cv::hconcat(debugImg, oldMask, debugImg);
	cv::imshow("debugImg", debugImg);

	cv::Mat fgAndCandid(frameSize, CV_8UC1, 0.0);
	fgAndCandid.setTo(128, cv::Mat(fgCandidate == 0 & fgMaskBlur == 0));
	fgAndCandid.setTo(0, cv::Mat(fgCandidate == 255));
	fgAndCandid.setTo(255, cv::Mat(fgMaskBlur == 255));
	cv::imshow("fgAndCandid", fgAndCandid);

	cv::Mat markers;
	fgAndCandid.convertTo(markers, CV_32S);

	cv::watershed(colorFrame, markers);
	cv::Mat mark;
	markers.convertTo(mark, CV_8U);

	fgMask.setTo(255, cv::Mat(mark == 255));

	

	//-- bounding box
	//cv::Mat fgMaskDilate;
	//cv::dilate(fgMask, fgMaskDilate, cv::Mat(), cv::Point(-1, -1), 16, 1, 1);
	//cv::erode(fgMask, fgMaskDilate, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);
	//cv::dilate(fgMask, fgMaskDilate, cv::Mat(), cv::Point(-1, -1), 4, 1, 1);

	//cv::imshow("fgMaskDilate", fgMaskDilate);

	//cv::RNG rng(12345);
	//std::vector<std::vector<cv::Point> > contours;
	////cv::findContours(fgMaskDilate, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
	//contours = findBiggestContour(fgMaskDilate, 1);
	//std::vector<std::vector<cv::Point> > contours_poly(contours.size());
	//std::vector<cv::Rect> boundRect(contours.size());
	//std::vector<cv::Point2f>centers(contours.size());
	//std::vector<cv::RotatedRect> minEllipse(contours.size());
	//std::vector<float>radius(contours.size());
	//for (size_t i = 0; i < contours.size(); i++)
	//{
	//	approxPolyDP(contours[i], contours_poly[i], 3, true);
	//	boundRect[i] = boundingRect(contours_poly[i]);
	//	minEnclosingCircle(contours_poly[i], centers[i], radius[i]);
	//	//minEllipse[i] = fitEllipse(contours[i]);
	//}
	//cv::Mat drawing = cv::Mat::zeros(fgMaskDilate.size(), CV_8UC3);
	//frame.copyTo(drawing);
	//for (size_t i = 0; i < contours.size(); i++)
	//{
	//	if (contours[i].size() > 10) {
	//		//cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
	//		cv::Scalar color = cv::Scalar(0, 0, 255);
	//		//drawContours(drawing, contours_poly, (int)i, color);
	//		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2);
	//		//circle(drawing, centers[i], (int)radius[i], color, 2);
	//		//circle(drawing, centers[i], 70, cv::Scalar(100, 0, 255), 2);
	//		//ellipse(drawing, minEllipse[i], color, 2);
	//	}
	//}
	//
	//drawing.copyTo(debugImg);
	//imshow("debugImg", debugImg);

#endif

	frameGray.copyTo(prevGray);
	bg = cv::Mat(modelHeight, modelWidth, CV_32F, m_Mean[0]);
	cv::resize(bg, bg, cv::Size(), BLOCK_SIZE, BLOCK_SIZE);
	bg.convertTo(bg, CV_8U);
}

//------------------------
//-- Gaussian model
//------------------------
void DCFG::motionCompensate(double h[9]) {

	//-- compensate models for the current view
	for (int j = 0; j < modelHeight; ++j) {
		for (int i = 0; i < modelWidth; ++i) {

			//-- x and y coordinates for current model (center of each block)
			float X, Y;
			float W = 1.0;
			X = BLOCK_SIZE * i + BLOCK_SIZE / 2.0;
			Y = BLOCK_SIZE * j + BLOCK_SIZE / 2.0;

			//-- transformed coordinates with h
			float newW = h[6] * X + h[7] * Y + h[8];
			float newX = (h[0] * X + h[1] * Y + h[2]) / newW;
			float newY = (h[3] * X + h[4] * Y + h[5]) / newW;

			//-- transformed i,j coordinates of old position
			float newI = newX / BLOCK_SIZE;
			float newJ = newY / BLOCK_SIZE;

			int idxNewI = floor(newI);
			int idxNewJ = floor(newJ);

			float di = newI - ((float)(idxNewI)+0.5);
			float dj = newJ - ((float)(idxNewJ)+0.5);

			float w_H{ 0.0 }, w_V{ 0.0 }, w_HV{ 0.0 }, w_self{ 0.0 }, sumW{ 0.0 };

			int idxNow = i + j * modelWidth;

#define WARP_MIX
			//-- For Mean and Age
			{
				float temp_mean[4][NUM_MODELS] = {};
				float temp_age[4][NUM_MODELS] = {};

#ifdef WARP_MIX
				//-- Horizontal Neighbor
				if (di != 0) {
					int idx_new_i = idxNewI;
					int idx_new_j = idxNewJ;
					idx_new_i += di > 0 ? 1 : -1;
					if (idx_new_i >= 0 && idx_new_i < modelWidth && idx_new_j >= 0 && idx_new_j < modelHeight) {
						w_H = fabs(di) * (1.0 - fabs(dj)); //-- intersected area
						sumW += w_H;
						int idxNew = idx_new_i + idx_new_j * modelWidth;
						for (int m = 0; m < NUM_MODELS; ++m) {
							temp_mean[0][m] = w_H * m_Mean[m][idxNew];
							temp_age[0][m] = w_H * m_Age[m][idxNew];
						}
					}
				}
				//-- Vertical Neighbor
				if (dj != 0) {
					int idx_new_i = idxNewI;
					int idx_new_j = idxNewJ;
					idx_new_j += dj > 0 ? 1 : -1;
					if (idx_new_i >= 0 && idx_new_i < modelWidth && idx_new_j >= 0 && idx_new_j < modelHeight) {
						w_V = fabs(dj) * (1.0 - fabs(di));
						sumW += w_V;
						int idxNew = idx_new_i + idx_new_j * modelWidth;
						for (int m = 0; m < NUM_MODELS; ++m) {
							temp_mean[1][m] = w_V * m_Mean[m][idxNew];
							temp_age[1][m] = w_V * m_Age[m][idxNew];
						}
					}
				}
				//-- HV Neighbor
				if (dj != 0 && di != 0) {
					int idx_new_i = idxNewI;
					int idx_new_j = idxNewJ;
					idx_new_i += di > 0 ? 1 : -1;
					idx_new_j += dj > 0 ? 1 : -1;
					if (idx_new_i >= 0 && idx_new_i < modelWidth && idx_new_j >= 0 && idx_new_j < modelHeight) {
						w_HV = fabs(di) * fabs(dj);
						sumW += w_HV;
						int idxNew = idx_new_i + idx_new_j * modelWidth;
						for (int m = 0; m < NUM_MODELS; ++m) {
							temp_mean[2][m] = w_HV * m_Mean[m][idxNew];
							temp_age[2][m] = w_HV * m_Age[m][idxNew];
						}
					}
				}
#endif
				//-- Self
				if (idxNewI >= 0 && idxNewI < modelWidth && idxNewJ >= 0 && idxNewJ < modelHeight) {
					w_self = (1.0 - fabs(di)) * (1.0 - fabs(dj));
					sumW += w_self;
					int idxNew = idxNewI + idxNewJ * modelWidth;
					for (int m = 0; m < NUM_MODELS; ++m) {
						temp_mean[3][m] = w_self * m_Mean[m][idxNew];
						temp_age[3][m] = w_self * m_Age[m][idxNew];
					}
				}

				if (sumW > 0) {
					for (int m = 0; m < NUM_MODELS; ++m) {
#ifdef WARP_MIX
						m_Mean_Temp[m][idxNow] = (temp_mean[0][m] + temp_mean[1][m] + temp_mean[2][m] + temp_mean[3][m]) / sumW;
						m_Age_Temp[m][idxNow] = (temp_age[0][m] + temp_age[1][m] + temp_age[2][m] + temp_age[3][m]) / sumW;
#else
						m_Mean_Temp[m][idxNow] = temp_mean[3][m] / sumW;
						m_Age_Temp[m][idxNow] = temp_age[3][m] / sumW;
#endif
					}
				}
			}


			//-- For Variance
			{
				float temp_var[4][NUM_MODELS];
				memset(temp_var, 0, sizeof(float) * 4 * NUM_MODELS);
#ifdef WARP_MIX
				//-- Horizontal Neighbor
				if (di != 0) {
					int idx_new_i = idxNewI;
					int idx_new_j = idxNewJ;
					idx_new_i += di > 0 ? 1 : -1;
					if (idx_new_i >= 0 && idx_new_i < modelWidth && idx_new_j >= 0 && idx_new_j < modelHeight) {
						int idxNew = idx_new_i + idx_new_j * modelWidth;
						for (int m = 0; m < NUM_MODELS; ++m) {
							temp_var[0][m] = w_H * (m_Var[m][idxNew] + VARIANCE_INTERPOLATE_PARAM * (pow(m_Mean_Temp[m][idxNow] - m_Mean[m][idxNew], (int)2)));
						}
					}
				}
				//-- Vertical Neighbor
				if (dj != 0) {
					int idx_new_i = idxNewI;
					int idx_new_j = idxNewJ;
					idx_new_j += dj > 0 ? 1 : -1;
					if (idx_new_i >= 0 && idx_new_i < modelWidth && idx_new_j >= 0 && idx_new_j < modelHeight) {
						int idxNew = idx_new_i + idx_new_j * modelWidth;
						for (int m = 0; m < NUM_MODELS; ++m) {
							temp_var[1][m] = w_V * (m_Var[m][idxNew] + VARIANCE_INTERPOLATE_PARAM * (pow(m_Mean_Temp[m][idxNow] - m_Mean[m][idxNew], (int)2)));
						}
					}
				}
				// HV Neighbor
				if (dj != 0 && di != 0) {
					int idx_new_i = idxNewI;
					int idx_new_j = idxNewJ;
					idx_new_i += di > 0 ? 1 : -1;
					idx_new_j += dj > 0 ? 1 : -1;
					if (idx_new_i >= 0 && idx_new_i < modelWidth && idx_new_j >= 0 && idx_new_j < modelHeight) {
						int idxNew = idx_new_i + idx_new_j * modelWidth;
						for (int m = 0; m < NUM_MODELS; ++m) {
							temp_var[2][m] = w_HV * (m_Var[m][idxNew] + VARIANCE_INTERPOLATE_PARAM * (pow(m_Mean_Temp[m][idxNow] - m_Mean[m][idxNew], (int)2)));
						}
					}
				}
#endif
				//-- Self
				if (idxNewI >= 0 && idxNewI < modelWidth && idxNewJ >= 0 && idxNewJ < modelHeight) {
					int idxNew = idxNewI + idxNewJ * modelWidth;
					for (int m = 0; m < NUM_MODELS; ++m) {
						temp_var[3][m] = w_self * (m_Var[m][idxNew] + VARIANCE_INTERPOLATE_PARAM * (pow(m_Mean_Temp[m][idxNow] - m_Mean[m][idxNew], (int)2)));
					}
				}

				if (sumW > 0) {
					for (int m = 0; m < NUM_MODELS; ++m) {
#ifdef WARP_MIX
						m_Var_Temp[m][idxNow] = (temp_var[0][m] + temp_var[1][m] + temp_var[2][m] + temp_var[3][m]) / sumW;
#else
						m_Var_Temp[m][idxNow] = (temp_var[3][m]) / sumW;
#endif
					}
				}

			}

			//-- Limitations and Exceptions
			for (int m = 0; m < NUM_MODELS; ++m) {
				m_Var_Temp[m][i + j * modelWidth] = MAX(m_Var_Temp[m][i + j * modelWidth], MIN_BG_VAR);
			}
			if (idxNewI < 1 || idxNewI >= modelWidth - 1 || idxNewJ < 1 || idxNewJ >= modelHeight - 1) {
				for (int m = 0; m < NUM_MODELS; ++m) {
					m_Var_Temp[m][i + j * modelWidth] = INIT_BG_VAR;
					m_Age_Temp[m][i + j * modelWidth] = 0;
				}
			}
			else {
				for (int m = 0; m < NUM_MODELS; ++m) {
					m_Age_Temp[m][i + j * modelWidth] =
						MIN(m_Age_Temp[m][i + j * modelWidth] * exp(-VAR_DEC_RATIO * MAX(0.0, m_Var_Temp[m][i + j * modelWidth] - VAR_MIN_NOISE_T)), MAX_BG_AGE);
				}
			}


			//-- debug
			//----------------------------------------
			//----------------------------------------

			//if (i * BLOCK_SIZE == 240 && j * BLOCK_SIZE == 240) {
			//	cv::Rect r = cv::Rect(X, Y, BLOCK_SIZE, BLOCK_SIZE);
			//	cv::Rect r2 = cv::Rect(newX, newY, BLOCK_SIZE, BLOCK_SIZE);
			//	cv::rectangle(prevColorFrame, r, cv::Scalar(255, 0, 0), 1);
			//	cv::rectangle(colorFrame, r2, cv::Scalar(0, 0, 255), 1);
			//	cv::putText(prevColorFrame, "Mean[0]: " + std::to_string(m_Mean[0][idxNow]), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 3);
			//	cv::putText(prevColorFrame, "Mean[0]: " + std::to_string(m_Mean[0][idxNow]), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 2);
			//	cv::putText(colorFrame, "Mean[1]: " + std::to_string(m_Mean[1][idxNow]), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 3);
			//	cv::putText(colorFrame, "Mean[1]: " + std::to_string(m_Mean[1][idxNow]), cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 2);
			//	cv::putText(prevColorFrame, "Age[0]: " + std::to_string(m_Age[0][idxNow]), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 3);
			//	cv::putText(prevColorFrame, "Age[0]: " + std::to_string(m_Age[0][idxNow]), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 2);
			//	cv::putText(colorFrame, "Age[1]: " + std::to_string(m_Age[1][idxNow]), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 3);
			//	cv::putText(colorFrame, "Age[1]: " + std::to_string(m_Age[1][idxNow]), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255), 2);

			//	cv::hconcat(prevColorFrame, colorFrame, debugImg);

			//	std::cout << "\n====================================\n" <<
			//		"m_Mean[0] = " << m_Mean[0][idxNow] << "   m_Mean_Temp[0]: " << m_Mean_Temp[0][idxNow] <<
			//		"   m_Var[0]: " << m_Var[0][idxNow] << "   m_Var_Temp[0]: " << m_Var_Temp[0][idxNow] <<
			//		"   m_Age[0]: " << m_Age[0][idxNow] << "   m_Age_Temp[0]: " << m_Age_Temp[0][idxNow] << std::endl <<
			//		"m_Mean[1] = " << m_Mean[1][idxNow] << "   m_Mean_Temp[1]: " << m_Mean_Temp[1][idxNow] <<
			//		"   m_Var[1]: " << m_Var[1][idxNow] << "   m_Var_Temp[1]: " << m_Var_Temp[1][idxNow] <<
			//		"   m_Age[1]: " << m_Age[1][idxNow] << "   m_Age_Temp[1]: " << m_Age_Temp[1][idxNow] << std::endl;

			//	outFile << "\n====================================\n" << 
			//		"m_Mean[0] = " << m_Mean[0][idxNow] << "   m_Mean_Temp[0]: " << m_Mean_Temp[0][idxNow] <<
			//		"   m_Var[0]: " << m_Var[0][idxNow] << "   m_Var_Temp[0]: " << m_Var_Temp[0][idxNow] <<
			//		"   m_Age[0]: " << m_Age[0][idxNow] << "   m_Age_Temp[0]: " << m_Age_Temp[0][idxNow] << std::endl <<
			//		"m_Mean[1] = " << m_Mean[1][idxNow] << "   m_Mean_Temp[1]: " << m_Mean_Temp[1][idxNow] <<
			//		"   m_Var[1]: " << m_Var[1][idxNow] << "   m_Var_Temp[1]: " << m_Var_Temp[1][idxNow] <<
			//		"   m_Age[1]: " << m_Age[1][idxNow] << "   m_Age_Temp[1]: " << m_Age_Temp[1][idxNow] << std::endl;
			//}
			//----------------------------------------
			//----------------------------------------
			//----------------------------------------

		}
	}
}


void DCFG::update(cv::Mat& pOutputImg) {
	pOutputImg = cv::Mat(frameSize, CV_8UC1, 0.0);
	uchar* pOut = (uchar*)pOutputImg.data;
	uchar* pOldOut = (uchar*)oldMask.data;
	uchar* pCur = (uchar*)frameGray.data;
	uchar* pDilatedFG = (uchar*)fgMaskDilate.data;
	uchar* pCandidateFG = (uchar*)fgCandidate.data;

	//////////////////////////////////////////////////////////////////////////
	//-- Find Matching Model
	for (int bIdx_j = 0; bIdx_j < modelHeight; bIdx_j++) {
		for (int bIdx_i = 0; bIdx_i < modelWidth; bIdx_i++) {
			int curIndex = bIdx_i + bIdx_j * modelWidth;

			//-- base (i,j) for this block in the frame
			int idx_base_i = ((float)bIdx_i) * BLOCK_SIZE;
			int idx_base_j = ((float)bIdx_j) * BLOCK_SIZE;

			float cur_mean = 0;
			float elem_cnt = 0;
			for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
				for (int ii = 0; ii < BLOCK_SIZE; ++ii) {

					int idx_i = idx_base_i + ii; //-- corresponding position of bg image in the frame
					int idx_j = idx_base_j + jj;

					if (idx_i < 0 || idx_i >= obsWidth || idx_j < 0 || idx_j >= obsHeight)
						continue;

					cur_mean += pCur[idx_i + idx_j * obsWidth];
					elem_cnt += 1.0;
				}
			}	//loop for pixels
			cur_mean /= elem_cnt; //-- mean of the pixels in the frame which correspond to the current block


			//////////////////////////////////////////////////////////////////////////
			//-- Make Oldest Idx to 0 (swap)
			int oldIdx = 0;
			float oldAge = 0;

			//-- find the oldest model 
			for (int m = 0; m < NUM_MODELS; ++m) {
				float fAge = m_Age_Temp[m][curIndex];

				if (fAge >= oldAge) {
					oldIdx = m;
					oldAge = fAge;
				}
			}

			//-- if there is an older model than the current model (model 0), swap them
			if (oldIdx != 0) {
				m_Mean_Temp[0][curIndex] = m_Mean_Temp[oldIdx][curIndex];
				m_Mean_Temp[oldIdx][curIndex] = cur_mean;

				m_Var_Temp[0][curIndex] = m_Var_Temp[oldIdx][curIndex];
				m_Var_Temp[oldIdx][curIndex] = INIT_BG_VAR;

				m_Age_Temp[0][curIndex] = m_Age_Temp[oldIdx][curIndex];
				m_Age_Temp[oldIdx][curIndex] = 0;
			}
			//////////////////////////////////////////////////////////////////////////
			// Select Model 
			// Check Match against 0
			if (pow(cur_mean - m_Mean_Temp[0][curIndex], (int)2) < VAR_THRESH_MODEL_MATCH * m_Var_Temp[0][curIndex]) {
				m_ModelIdx[curIndex] = 0;
			}
			// Check Match against 1
			else if (pow(cur_mean - m_Mean_Temp[1][curIndex], (int)2) < VAR_THRESH_MODEL_MATCH * m_Var_Temp[1][curIndex]) {
				m_ModelIdx[curIndex] = 1;
			}
			// If No match, set 1 age to zero and match = 1
			else {

				//-- update foreground model
				int count = 0;
				unsigned char pixel = m_Mean_Temp[1][curIndex];
				// Let the pixel compare with the backgroundModel set
				for (std::vector<cv::Mat>::iterator it = backgroundModel.begin(); it != backgroundModel.end(); it++)
					if (abs(int(pixel) - int((*it).at<uchar>(bIdx_j, bIdx_i))) < distanceThreshold)
						count++;
				if (count < this->minMatch)
					updateFG(m_Mean_Temp[1][curIndex]);

				//-- initialize the candid bg model
				m_ModelIdx[curIndex] = 1;
				m_Age_Temp[1][curIndex] = 0;
			}

#if DFCG_DEBUG
			//int debugX = 240;
			//int debugY = 240;
			//int n = 1;
			//cv::Mat myImg;
			//colorFrame.copyTo(myImg);

			//if (idx_base_i == debugX && idx_base_j == debugY) {
			//	std::cout << "\ncur_mean: " << cur_mean << std::endl;

			//	for (int i = -n; i <= n; i++) {
			//		for (int j = -n; j <= n; j++) {
			//			int idX = debugX + i * BLOCK_SIZE;
			//			int idY = debugY + j * BLOCK_SIZE;
			//			int curIndex1 = (bIdx_i + i) + (bIdx_j + j) * modelWidth;
			//			cv::Rect rect(idX, idY, BLOCK_SIZE, BLOCK_SIZE);
			//			cv::rectangle(myImg, rect, cv::Scalar(255, 0, 0));
			//			std::cout << "idX: " << idX << "\tidY: " << idY << "\tm_Mean_Temp[0]: " << m_Mean_Temp[0][curIndex1] << "\tm_Mean_Age[0]: " << m_Age_Temp[0][curIndex1]
			//				<< "\tm_Mean_Temp[1]: " << m_Mean_Temp[1][curIndex1] << "\tm_Mean_Age[1]: " << m_Age_Temp[1][curIndex1] << std::endl;
			//		}
			//	}

			//	

			//	cv::imshow("myImg", myImg);
			//}
			//
#endif

		}
	}		//-- loop for models

	//-- update with current observation
	float obs_mean[NUM_MODELS];
	float obs_var[NUM_MODELS];

	//-- update mean
	for (int bIdx_j = 0; bIdx_j < modelHeight; bIdx_j++) {
		for (int bIdx_i = 0; bIdx_i < modelWidth; bIdx_i++) {
			int curIndex = bIdx_i + bIdx_j * modelWidth;

			// base (i,j) for this block
			int idx_base_i = ((float)bIdx_i) * BLOCK_SIZE;
			int idx_base_j = ((float)bIdx_j) * BLOCK_SIZE;

			int nMatchIdx = m_ModelIdx[curIndex]; //-- the model it has been matched with (0 or 1)

			//-- obtain observation mean
			memset(obs_mean, 0, sizeof(float) * NUM_MODELS);
			int nElemCnt[NUM_MODELS] = {};
			for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
				for (int ii = 0; ii < BLOCK_SIZE; ++ii) {

					int idx_i = idx_base_i + ii;
					int idx_j = idx_base_j + jj;

					if (idx_i < 0 || idx_i >= obsWidth || idx_j < 0 || idx_j >= obsHeight)
						continue;

					obs_mean[nMatchIdx] += pCur[idx_i + idx_j * obsWidth];
					++nElemCnt[nMatchIdx];
				}
			}
			for (int m = 0; m < NUM_MODELS; ++m) {

				if (nElemCnt[m] <= 0) {
					m_Mean[m][curIndex] = m_Mean_Temp[m][curIndex];
				}
				else {
					//-- learning rate for this block
					float age = m_Age_Temp[m][curIndex];
					float alpha = age / (age + 1.0);

					obs_mean[m] /= ((float)nElemCnt[m]);
					//-- update with this mean
					if (age < 1) {
						m_Mean[m][curIndex] = illum + obs_mean[m];
					}
					else if (blockMag.at<float>(idx_base_j, idx_base_i) < BLOCK_SIZE)
						m_Mean[m][curIndex] = illum + obs_mean[m];
					else 
						m_Mean[m][curIndex] = alpha * (illum + m_Mean_Temp[m][curIndex]) + (1.0 - alpha) * obs_mean[m];

				}
			}

			bgMean += m_Mean[0][curIndex];
		}
	}

	//-- update variance & classify
	for (int bIdx_j = 0; bIdx_j < modelHeight; bIdx_j++) {
		for (int bIdx_i = 0; bIdx_i < modelWidth; bIdx_i++) {
			int curIndex = bIdx_i + bIdx_j * modelWidth;
			// TODO: OPTIMIZE THIS PART SO THAT WE DO NOT CALCULATE THIS (LUT)
			// base (i,j) for this block
			int idx_base_i = ((float)bIdx_i) * BLOCK_SIZE;
			int idx_base_j = ((float)bIdx_j) * BLOCK_SIZE;

			int nMatchIdx = m_ModelIdx[curIndex];

			// obtain observation variance
			memset(obs_var, 0, sizeof(float) * NUM_MODELS);
			int nElemCnt[NUM_MODELS] = {};
			float* fgDist = new float[NUM_MODELS_FG];

			for (int jj = 0; jj < BLOCK_SIZE; ++jj) {
				for (int ii = 0; ii < BLOCK_SIZE; ++ii) {

					int idx_i = idx_base_i + ii;
					int idx_j = idx_base_j + jj;
					nElemCnt[nMatchIdx]++;

					if (idx_i < 0 || idx_i >= obsWidth || idx_j < 0 || idx_j >= obsHeight) {
						continue;
					}

					float pixelDist = 0.0;
					float fDiff = pCur[idx_i + idx_j * obsWidth] - m_Mean[nMatchIdx][curIndex];
					pixelDist += pow(fDiff, (int)2);


					//-- classification
					//-- pCur[idx_i + idx_j * obsWidth] : I_j in the paper section 2.4
					//-- m_Var_Temp[0][curIndex] : mu_ {A,i} in the paper section 2.4

					m_DistImg[idx_i + idx_j * obsWidth] = pow(pCur[idx_i + idx_j * obsWidth] - m_Mean[0][curIndex], (int)2);
					for (int i = 0; i < f_Num; i++) {
						m_DistImg_FG[i][idx_i + idx_j * obsWidth] = pow(pCur[idx_i + idx_j * obsWidth] - f_Mean[i], (int)2);
					}
					
					if (!pOutputImg.empty() && m_Age_Temp[0][curIndex] > 1) {

						uchar valOldOut = m_DistImg[idx_i + idx_j * obsWidth] > VAR_THRESH_BG_DETERMINE * m_Var_Temp[0][curIndex] ? 255 : 0;
						pOldOut[idx_i + idx_j * obsWidth] = valOldOut;

						//---------------------------------------------------------------

						//uchar valOut = m_DistImg[idx_i + idx_j * obsWidth] > m_DistImg_FG[idx_i + idx_j * obsWidth] ? 255 : 0;
						//pOut[idx_i + idx_j * obsWidth] = valOut;

						//---------------------------------------------------------------

						/*float bgDist = m_DistImg[idx_i + idx_j * obsWidth] - m_Var_Temp[0][curIndex];

						int count = 0;
						for (int i = 0; i < f_Num; i++) {
							if (bgDist < fgDist[i])
								count++;
						}

						uchar valOut = (count < 2) ? 255 : 0;
						pOut[idx_i + idx_j * obsWidth] = valOut;*/

						//---------------------------------------------------------------
						//uchar valOut = ((m_DistImg[idx_i + idx_j * obsWidth] > VAR_THRESH_BG_DETERMINE* m_Var_Temp[0][curIndex]) &&
						//	(m_DistImg_FG[0][idx_i + idx_j * obsWidth] < VAR_THRESH_BG_DETERMINE* f_Var[0]))? 255 : 0;
						//pOut[idx_i + idx_j * obsWidth] = valOut;

						//---------------------------------------------------------------

						//int count = 0;
						//for (int i = 0; i < f_Num; i++) {
						//	if (m_DistImg[idx_i + idx_j * obsWidth] < m_DistImg_FG[i][idx_i + idx_j * obsWidth])
						//		count++;
						//}

						//uchar valOut = (count > 1) ? 0 : 255;
						//pOut[idx_i + idx_j * obsWidth] = valOut;
						//---------------------------------------------------------------
						
						uchar valOut = 0;
						if (m_DistImg[idx_i + idx_j * obsWidth] > VAR_THRESH_BG_DETERMINE* m_Var_Temp[0][curIndex]) {
							pCandidateFG[idx_i + idx_j * obsWidth] = 255;
							for (int i = 0; i < f_Num / 2; i++) {
								//if (m_DistImg_FG[i][idx_i + idx_j * obsWidth] < VAR_THRESH_FG_DETERMINE * f_Var[i])
								if (m_DistImg_FG[i][idx_i + idx_j * obsWidth] < VAR_THRESH_FG_DETERMINE * m_Var_Temp[0][curIndex])
									valOut = 255;
							}
						}

						pOut[idx_i + idx_j * obsWidth] = valOut;
						//---------------------------------------------------------------


						//if (idx_base_i == 240 && idx_base_j == 240) {
						////if (idx_base_i == 220 && idx_base_j == 180) {

						//	std::cout << "pCur: " << (int)pCur[idx_i + idx_j * obsWidth]
						//		<< "\nm_Mean[0]: " << m_Mean[0][curIndex] << "\tm_Var_Temp[0]: " << m_Var_Temp[0][curIndex] 
						//		<< "\tm_DistImg = " << m_DistImg[idx_i + idx_j * obsWidth]  << "\tvalOut: " << (int)valOut
						//		<< "\n-------------------------------------------" << std::endl;

						//	//for (int i = 0; i < f_Num; i++) {
						//	//	std::cout 
						//	//		<< "fmean[" << i << "]: " << f_Mean[i] << "\tf_Var[" << i << "]: " << f_Var[i] 
						//	//		<< "\tf_Age[" << i << "]: " << f_Age[i] << "\tf_Weight[" << i << "]: " << f_Weight[i] << std::endl
						//	//		<< "m_DistImg_FG[" << i << "]: " << m_DistImg_FG[i][idx_i + idx_j * obsWidth] 
						//	//		<< "\n-------------------------------------------" << std::endl;
						//	//}

						//	std::cout << "\n=================================================================\n";
						//		
						//}
					}

					obs_var[nMatchIdx] = MAX(obs_var[nMatchIdx], pixelDist);


					//-- debug
					//----------------------------------------------
					//if (idx_base_i == 240 && idx_base_j == 240) {
					//if (idx_base_i == 220 && idx_base_j == 180) {

						//cv::Mat testBlocks = prevGray.clone();
						//cv::Rect r = cv::Rect(idx_base_i, idx_base_j, BLOCK_SIZE, BLOCK_SIZE);
						//cv::rectangle(testBlocks, r, cv::Scalar(255), 3);
						//cv::imshow("testBlocks", testBlocks);


						//std::cout << "m_DistImg = " << m_DistImg[idx_i + idx_j * obsWidth] << "\tm_Var_Temp[0]: " << m_Var_Temp[0][curIndex] <<
						//	"\tm_Mean[0]: " << m_Mean[0][curIndex] << "\tpCur: " << (int)pCur[idx_i + idx_j * obsWidth] << "\tpOut: " << (int)pOut[idx_i + idx_j * obsWidth] <<
						//	"\tm_DistImg_FG = " << m_DistImg_FG[idx_i + idx_j * obsWidth] <<
						//	"\n-------------------------------------------" << std::endl;

						//outFile << "m_DistImg = " << m_DistImg[idx_i + idx_j * obsWidth] << "\tm_Var_Temp[0]: " << m_Var_Temp[0][curIndex] <<
						//	"\tm_Mean[0]: " << m_Mean[0][curIndex] << "\tpCur: " << (int)pCur[idx_i + idx_j * obsWidth] << "\tpOut: " << (int)pOut[idx_i + idx_j * obsWidth] <<
						//	"\n-------------------------------------------" << std::endl;
					//}
					//----------------------------------------------
					//----------------------------------------------

				}
			}


			for (int m = 0; m < NUM_MODELS; ++m) {

				if (nElemCnt[m] > 0) {
					float age = m_Age_Temp[m][curIndex];
					float alpha = age / (age + 1.0);

					//-- update with this variance
					if (age == 0) {
						m_Var[m][curIndex] = MAX(obs_var[m], INIT_BG_VAR);
					}
					else {
						float alpha_var = alpha;	//MIN(alpha, 1.0 - MIN_NEW_VAR_OBS_PORTION);
						m_Var[m][curIndex] = alpha_var * m_Var_Temp[m][curIndex] + (1.0 - alpha_var) * obs_var[m];
						m_Var[m][curIndex] = MAX(m_Var[m][curIndex], MIN_BG_VAR);
					}

					//-- Update Age
					m_Age[m][curIndex] = MIN(m_Age_Temp[m][curIndex] + 1.0, MAX_BG_AGE);
				}
				else {
					m_Var[m][curIndex] = m_Var_Temp[m][curIndex];
					m_Age[m][curIndex] = m_Age_Temp[m][curIndex];
				}
			}


		}
	}

	//std::cout << "\nvalout: " << (int)pOutputImg.at<uchar>(240, 240);
}



//------------------------
//-- KLT
//------------------------
void DCFG::KLT(cv::Mat& imgGray) {
#if DFCG_DEBUG
	cv::RNG rng(12345);
	colorFrame.copyTo(testImg);
#endif

	if (!points[0].empty())
	{
		calcOpticalFlowPyrLK(prevGray, imgGray, points[0], points[1], status, err, winSize, 3, termcrit, 0, 0.001);
		size_t i;
		for (i = count = 0; i < points[1].size(); i++)
		{
			if (!status[i])
				continue;
			nMatch[count++] = i;
#if DFCG_DEBUG
			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			circle(testImg, points[0][i], 3, color, -1, 8);
			circle(testImg, points[1][i], 3, color, -1, 8);
			cv::arrowedLine(testImg, points[0][i], points[1][i], color, 1);
#endif
		}
	}


	//---------------------------------------------
	//estimate Optical Flow
	//cv::Mat flow, ang, mag, hsv, rgb;
	//std::vector<cv::Mat> flowChannels(2);
	//cv::calcOpticalFlowFarneback(prevGray, imgGray, flow, 0.5, 3, 21, 20, 5, 1.1,0);

	//cv::add(flow, cumulativeFlow, cumulativeFlow);

	//flowSeq.push_back(flow.clone());
	//if (flowSeq.size() > 5) {
	//	cv::subtract(cumulativeFlow, flowSeq.front(), cumulativeFlow);
	//	flowSeq.pop_front();
	//}

	//frameSeq.push_back(imgGray.clone());
	//if (frameSeq.size() > 5) 
	//	frameSeq.pop_front();


	//cv::split(cumulativeFlow, flowChannels);

	//cv::cartToPolar(flowChannels[0], flowChannels[1], mag, ang, false);

	//cv::Mat sat = cv::Mat(imgGray.rows, imgGray.cols, CV_8UC1, 255.0);
	//cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX, -1, cv::noArray());
	//ang = ang * 180 / CV_PI / 2;
	////double  minVal, maxVal;
	////cv::minMaxLoc(ang, &minVal, &maxVal);  //find  minimum  and  maximum  intensities
	////ang.convertTo(ang, CV_8U, 255.0 / (maxVal - minVal), -minVal);
	//mag.convertTo(mag, CV_8U);
	//ang.convertTo(ang, CV_8U);


	//std::vector<cv::Mat> hsvChannels;
	//hsvChannels.push_back(ang);
	//hsvChannels.push_back(sat);
	//hsvChannels.push_back(mag);

	//cv::merge(hsvChannels, hsv);

	//cv::cvtColor(hsv, rgb, CV_HSV2BGR);
	//cv::imshow("rgb", rgb);
	//cv::imshow("h", hsvChannels[0]);
	//cv::imshow("s", hsvChannels[1]);
	//cv::imshow("v", hsvChannels[2]);


	//for (int y = 0; y < prevGray.rows; y ++) {
	//	for (int x = 0; x < prevGray.cols; x ++) {
	//		//const cv::Point2f flowatxy = cumulativeFlow.at<cv::Point2f>(y, x);
	//		//ang.at<float>(y,x) = (atan2(flowatxy.y, flowatxy.x) + CV_PI) * (180 / (float)CV_PI);
	//		std::cout << (int)ang.at<uchar>(y, x) << " ";
	//		//mag.at<float>(y, x) = sqrt(pow(flowatxy.x, 2) + pow(flowatxy.y, 2)) * 10;
	//		std::cout << (int)mag.at<uchar>(y, x) << std::endl;
	//	}
	//}

	//ang.convertTo(ang, CV_8UC1);
	//mag.convertTo(mag, CV_8UC1);

	//cv::imshow("ang", ang);
	//cv::imshow("mag", mag);

	//-------------------- draw optical flow vectors
	//cv::Mat motionSeg = colorFrame.clone();
	//for (int y = 0; y < prevGray.rows; y += 10) {
	//	for (int x = 0; x < prevGray.cols; x += 10) {
	//		// get the flow from y, x position * 10 for better visibility
	//		const cv::Point2f flowatxy = cumulativeFlow.at<cv::Point2f>(y, x) * 10;
	//		// draw line at flow direction
	//		cv::line(motionSeg, cv::Point(x, y), cv::Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), cv::Scalar(25 * (int)flowatxy.x, 25 * (int)flowatxy.y, 0));
	//		// draw initial point
	//		cv::circle(motionSeg, cv::Point(x, y), 1, cv::Scalar(0, 0, 0), -1);
	//	}
	//}
	//// draw the results
	//cv::namedWindow("motionSeg", cv::WINDOW_AUTOSIZE);
	//cv::imshow("motionSeg", motionSeg);

	////-------------------- segment
	//cv::Mat segFlow = cumulativeFlow.clone();
	//segFlow = segFlow.reshape(1, segFlow.total());

	//cv::Mat labels, centers;
	//int K = 3;
	//cv::TermCriteria tc;

	//kmeans(segFlow, K, labels, cv::TermCriteria(CV_TERMCRIT_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

	//// replace pixel values with their center value:
	//cv::Vec2f* p = segFlow.ptr<cv::Vec2f>();
	//for (size_t i = 0; i < segFlow.cols; i++) {
	//	int center_id = labels.at<int>(i);
	//	p[i] = centers.at<cv::Vec2f>(center_id);
	//}

	////--------------------------------------------------------
	//cv::Mat motionSeg = colorFrame.clone();
	//for (int y = 0; y < prevGray.rows; y += 5) {
	//	for (int x = 0; x < prevGray.cols; x += 5) {
	//		// get the segFlow from y, x position * 10 for better visibility
	//		//const cv::Point2f flowatxy = segFlow.at<cv::Point2f>(y, x) * 10;
	//		const cv::Point2f flowatxy = segFlow.at<cv::Vec2f>(y*x) * 5;
	//		// draw line at segFlow direction
	//		cv::line(motionSeg, cv::Point(x, y), cv::Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), cv::Scalar(25 * (int)flowatxy.x, 25 * (int)flowatxy.y, 25 * (int)flowatxy.x));
	//		// draw initial point
	//		cv::circle(motionSeg, cv::Point(x, y), 1, cv::Scalar(0, 0, 0), -1);
	//	}
	//}

	////--------------------------------------------------------

	//// draw the results
	//cv::namedWindow("motionSeg", cv::WINDOW_AUTOSIZE);
	//cv::imshow("motionSeg", motionSeg);

	//cv::Mat motionSeg2;
	//cv::Mat labels2(labels.size(), CV_8UC3, 0.0);
	//for (int i = 0; i < labels.rows; i++)
	//	switch (labels.at<int>(i)) {
	//	case 0:
	//		labels2.at<cv::Vec3b>(i) = { 255, 0, 0 };
	//		break;
	//	case 1:
	//		labels2.at<cv::Vec3b>(i) = { 0, 255, 0 };
	//		break;
	//	case 2:
	//		labels2.at<cv::Vec3b>(i) = { 0, 0, 255 };
	//		break;
	//	case 3:
	//		labels2.at<cv::Vec3b>(i) = { 255, 0, 255 };
	//		break;
	//	case 4:
	//		labels2.at<cv::Vec3b>(i) = { 0, 255, 255 };
	//		break;
	//	default:
	//		labels2.at<cv::Vec3b>(i) = { 0, 0, 0 };
	//	}
	//motionSeg2 = labels2.reshape(0, prevGray.rows);
	//cv::namedWindow("motionSeg2", cv::WINDOW_AUTOSIZE);
	//cv::imshow("motionSeg2", colorFrame + .5 * motionSeg2);

	//---------------------------------------------
	// 3.1 Step 3: Camera/background motion estimation 
			// pointsPrev, pointsCur are vector<Point2f> from LK

	//std::vector<cv::Point2f> foreground;
	//std::vector<double> D;
	//if (count >= 10) {
	//	cv::Mat H = cv::findHomography(points[0], points[1], CV_RANSAC);
	//	//std::cerr << H << std::endl;
	//	// now, backproject the points, and see, how far off they are:
	//	for (size_t i = 0; i < points[1].size(); i++)
	//	{
	//		cv::Point2f p0 = points[0][i];
	//		cv::Point2f p1 = points[1][i];
	//		// homogeneous point for mult:
	//		cv::Mat_<double> col(3, 1);
	//		col << p0.x, p0.y, 1;
	//		col = H * col;
	//		col /= col(2); // divide by W
	//		double dist = sqrt(pow(col(0) - p1.x, 2) +
	//			pow(col(1) - p1.y, 2));
	//		D.push_back(dist);
	//		// small distance == inlier == camera motion
	//		// large distance == outlier == object motion
	//		//if (dist >= 1.5)  // some heuristical threshold value
	//		{
	//			foreground.push_back(p1);
	//		}
	//	}
	//	//std::cerr << "fg " << points[1].size() << " " << foreground.size() << std::endl;

	//	cv::Mat motionSeg(frameSize, CV_8UC3, 0.0);
	//	int myradius = 5;
	//	for (int i = 0; i < foreground.size(); i++)
	//		cv::circle(motionSeg, cvPoint(foreground[i].x, foreground[i].y), myradius*D[i], CV_RGB(100, 100, 0), -1, 8, 0);

	//	cv::imshow("motionSeg", motionSeg);
	//}
	//---------------------------------------------

#if DFCG_DEBUG
	imshow("LK Demo", testImg);
#endif

	//-- find homography
	if (count >= 10) {
		//-- Make homography matrix with correspondences
		std::vector < cv::Point2f > pt1, pt2;
		int i;
		pt1.resize(count);
		pt2.resize(count);
		for (i = 0; i < count; i++) {
			//REVERSE HOMOGRAPHY
			pt1[i] = points[1][nMatch[i]];
			pt2[i] = points[0][nMatch[i]];
		}

		cv::Mat homography = cv::findHomography(pt1, pt2, CV_RANSAC);
		std::memcpy(matH, homography.data, sizeof(double) * 9);

		//---------------------------------------------
		//-- Foreground motion estimation
		//-- based on paper: Scene conditional background update for moving object detection in a moving camera 2.1.2
		for (size_t i = 0; i < pt1.size(); i++)
		{
			cv::Point2f p0 = pt2[i];
			cv::Point2f p1 = pt1[i];
			// homogeneous point for mult:
			cv::Mat_<double> col(3, 1);
			col << p0.x, p0.y, 1;
			col = homography * col;
			col /= col(2); // divide by W
			float angle = (atan2(col(1) - p1.y, col(0) - p1.x) + CV_PI) * (180 / (float)CV_PI);
			float magnitute = sqrt(pow(col(0) - p1.x, 2) + pow(col(1) - p1.y, 2));
			blockMag(cv::Rect(p0.x - GRID_SIZE_W / 2, p0.y - GRID_SIZE_H / 2, GRID_SIZE_W, GRID_SIZE_H)) = magnitute;
			blockAng(cv::Rect(p0.x - GRID_SIZE_W / 2, p0.y - GRID_SIZE_H / 2, GRID_SIZE_W, GRID_SIZE_H)) = angle;
		}

		//ang.convertTo(ang, CV_8U, 1);
		//cv::imshow("ang", ang);

		//mag.convertTo(mag, CV_8U, 10);
		//cv::imshow("mag", mag);

		//-- estimate foregroung velocity
		fgPixelNum = 0;
		fgVelocity = 0;
		fgAngle = 0;
		for (int y = 0; y < obsHeight; y++) 
			for (int x = 0; x < obsWidth; x++) {
				if (fgMaskDilate.at<uchar>(y, x) == 255 && blockMag.at<float>(y, x) > 0 && blockAng.at<float>(y, x) > 0) {
					fgVelocity += blockMag.at<float>(y,x);
					fgAngle += blockAng.at<float>(y, x);
					fgPixelNum++;
				}
			}
		fgVelocity /= fgPixelNum;
		fgAngle /= fgPixelNum;

		//std::cout << "velocity: " << fgVelocity << "\tangle: " << fgAngle << "\tfgPixelNum: " << fgPixelNum << std::endl;

		//---------------------------------------------
		//std::vector<cv::Point2f> foreground;
		//std::vector<double> D;
		//if (count >= 10) {
		//	cv::Mat H = cv::findHomography(points[0], points[1], CV_RANSAC);
		//	//std::cerr << H << std::endl;
		//	// now, backproject the points, and see, how far off they are:
		//	for (size_t i = 0; i < points[1].size(); i++)
		//	{
		//		cv::Point2f p0 = points[0][i];
		//		cv::Point2f p1 = points[1][i];
		//		// homogeneous point for mult:
		//		cv::Mat_<double> col(3, 1);
		//		col << p0.x, p0.y, 1;
		//		col = H * col;
		//		col /= col(2); // divide by W
		//		double dist = sqrt(pow(col(0) - p1.x, 2) +
		//			pow(col(1) - p1.y, 2));
		//		D.push_back(dist);
		//		// small distance == inlier == camera motion
		//		// large distance == outlier == object motion
		//		if (dist >= 1.5)  // some heuristical threshold value
		//		{
		//			foreground.push_back(p1);
		//		}
		//	}
		//	//std::cerr << "fg " << points[1].size() << " " << foreground.size() << std::endl;

		//	cv::Mat motionSeg(frameSize, CV_8UC3, 0.0);
		//	int myradius = 5;
		//	for (int i = 0; i < foreground.size(); i++)
		//		cv::circle(motionSeg, cvPoint(foreground[i].x, foreground[i].y), myradius * D[i], CV_RGB(100, 100, 0), -1, 8, 0);

		//	cv::imshow("motionSeg", motionSeg);
		//}
		//---------------------------------------------

		//------------------------------------------------
		//-- debug
		//------------------------------------------------
		cv::Mat warp, warpColor, frameColor;
		cv::Mat mask(frameGray.size(), CV_8U, cv::Scalar(255));
		warpPerspective(prevGray, warp, homography, prevGray.size());
		warpPerspective(mask, mask, homography, mask.size());

		//cv::cvtColor(warp, warpColor, CV_GRAY2BGR);
		//cv::cvtColor(prevGray, frameColor, CV_GRAY2BGR);
		//imshow("warpColor", warpColor);
		//imshow("frameColor", frameColor);

		//------------------------------------------------
		//-- ViBe
		//------------------------------------------------
		cv::Mat smallFrame, rep;
		cv::resize(warp, smallFrame, cv::Size(), 1.0 / BLOCK_SIZE, 1.0 / BLOCK_SIZE);
		smallFrame.copyTo(rep);
		//frameGray.copyTo(rep);

		originalVibe_ClassifyAndUpdate_GRAY(smallFrame, seg);
		//cv::medianBlur(seg, seg, 5);
		//cv::imshow("segmentation", seg);

		//cv::Rect r = cv::Rect(60, 60, 1, 1);
		//cv::rectangle(rep, r, cv::Scalar(255), 3);
		//cv::imshow("rep", rep);

		//for (int i=0; i< numberSamples; i++)
			//cv::imshow("backgroundModel[" + std::to_string(i)+ "]", backgroundModel[i]);
		//cv::imshow("backgroundModel[0]", backgroundModel[0]);

		//std::cout << "\nbg set:\n";
		//for (int i = 0; i < backgroundModel.size(); i++) {
		//	std::cout << (int)backgroundModel[i].at<uchar>(60, 60) << "\t";
		//}
		//std::cout << "\n=======================================";


		//std::cout << "m_DistImg = " << m_DistImg[idx_i + idx_j * obsWidth] << "\tm_Var_Temp[0]: " << m_Var_Temp[0][curIndex] <<
		//	"\tm_Mean[0]: " << m_Mean[0][curIndex] << "\tpCur: " << (int)pCur[idx_i + idx_j * obsWidth] << "\tpOut: " << (int)pOut[idx_i + idx_j * obsWidth] <<
		//	"\tm_DistImg_FG = " << m_DistImg_FG[idx_i + idx_j * obsWidth] <<
		//	"\n-------------------------------------------" << std::endl;
		

		//------------------------------------------------
		////-- frame differencing
		//------------------------------------------------
		cv::Mat diff;
		//cv::absdiff(frameGray, warp, diff);
		cv::subtract(frameGray, warp, diff, fgMaskDilate);
		cv::imshow("frameGray", frameGray);
		cv::imshow("warp", warp);
		cv::imshow("diff", diff);
		//cv::imshow("testImg", testImg);
		//------------------------------------------------

	}
	else {
		for (int ii = 0; ii < 9; ++ii) {
			matH[ii] = ii % 3 == ii / 3 ? 1.0f : 0.0f;
		}
	}

	//-- init features (select points)
	points[1].clear();

	for (int i = 0; i < obsWidth / GRID_SIZE_W - 1; ++i) {
		for (int j = 0; j < obsHeight / GRID_SIZE_H - 1; ++j) {
			tempP.x = i * GRID_SIZE_W + GRID_SIZE_W / 2;
			tempP.y = j * GRID_SIZE_H + GRID_SIZE_H / 2;
			//if(fgMaskDilate.at<uchar>(tempP) == 0)
				points[1].push_back(tempP);
		}
	}

	//-- swap points
	std::swap(points[1], points[0]);


	

}



std::vector<std::vector<cv::Point>> findBiggestContour(cv::Mat& img, int numOfContours) {

	std::vector<std::vector<cv::Point> > contours;
	std::vector<std::vector<cv::Point>> biggest_contours;
	std::vector<cv::Vec4i > hierarchy;
	double minArea = 0.0;

	//-- crop sides of image
	const short crop_size = 3;
	img(cv::Range(0, crop_size), cv::Range(0, img.cols)) = 0; // crop top 10 pixels
	img(cv::Range(img.rows - crop_size, img.rows), cv::Range(0, img.cols)) = 0; // crop bottom 10 pixels
	img(cv::Range(0, img.rows), cv::Range(0, crop_size)) = 0; // crop down 10 pixels
	img(cv::Range(0, img.rows), cv::Range(img.cols - crop_size, img.cols)) = 0; // crop up 10 pixels

	cv::Mat thresh_img;
	//threshold(img, thresh_img, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	threshold(img, thresh_img, 20, 255, CV_THRESH_BINARY);
	//imshow("thresh_img", thresh_img);

	findContours(thresh_img.clone(), contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_L1, cv::Point());

	if (contours.size() > 0) {

		std::vector<double> areas;
		for (int i = 0; i < contours.size(); i++)
			areas.push_back(contourArea(contours[i])); // contours areas

		std::sort(areas.begin(), areas.end(), std::greater<double>());

		if (contours.size() < numOfContours) {
			for (int i = 0; i < contours.size(); i++) {
				if (contourArea(contours[i]) > areas[0] / 5)
				{
					biggest_contours.push_back(contours[i]);
				}
			}
		}
		else {
			for (int i = 0; i < contours.size(); i++) {
				if (contourArea(contours[i]) > areas[numOfContours] && contourArea(contours[i]) > areas[0] / 5)
				{
					biggest_contours.push_back(contours[i]);
				}
			}
		}
	}

	contours.clear();
	hierarchy.clear();

	return biggest_contours;
}


void DCFG::updateFG(float input) {
	float totalWeight = 0.0;

	//-----------------------------------------------------------------------
	//for (int i = 0; i < f_Num; i++) {
	//	std::cout
	//		<< "fmean[" << i << "]: " << f_Mean[i] << "\tf_Var[" << i << "]: " << f_Var[i]
	//		<< "\tf_Age[" << i << "]: " << f_Age[i] << "\tf_Weight[" << i << "]: " << f_Weight[i] << std::endl;
	//}
	//std::cout << "=================================================================\n";
	//std::cout << "input: " << input << std::endl;
	//std::cout << "=================================================================\n";
	//-----------------------------------------------------------------------


	if (f_Num == 0) {
		f_Var[0] = INIT_FG_VAR;
		f_Mean[0] = input;
		f_Age[0]++;
		f_Num++;
	}
	else if (f_Num < NUM_MODELS_FG) {
		//-- check if input fits in any of the existing models
		bool exist = false;
		for (int k = 0; k < f_Num; k++) {
			if (pow(input - f_Mean[k], (int)2) < VAR_THRESH_MODEL_MATCH * f_Var[k]) {
				exist = true;
				float var_temp = (f_Age[k] * f_Var[k] + (f_Mean[k] - input) * (f_Mean[k] - input)) / (f_Age[k] + 1);
				f_Var[k] = var_temp < MIN_FG_VAR ? MIN_FG_VAR : (var_temp > MAX_FG_VAR ? MAX_FG_VAR : var_temp);
				f_Mean[k] = (f_Age[k] * f_Mean[k] + input) / (f_Age[k] + 1);
				f_Age[k]++;
				break;
			}
		}
		//-- if input didn't match with any existing model, add a new model with the new input
		if (exist == false) {
			f_Var[f_Num] = INIT_FG_VAR;
			f_Mean[f_Num] = input;
			f_Age[f_Num]++;
			f_Num++;
		}
	}
	
	else {
		//-- check if input fits in any of the existing models
		bool exist = false;
		for (int k = 0; k < f_Num; k++) {
			if (pow(input - f_Mean[k], (int)2) < VAR_THRESH_MODEL_MATCH * f_Var[k]) {
				exist = true;
				float var_temp = (f_Age[k] * f_Var[k] + (f_Mean[k] - input) * (f_Mean[k] - input)) / (f_Age[k] + 1);
				f_Var[k] = var_temp < MIN_FG_VAR ? MIN_FG_VAR : (var_temp > MAX_FG_VAR ? MAX_FG_VAR : var_temp);
				f_Mean[k] = (f_Age[k] * f_Mean[k] + input) / (f_Age[k] + 1);
				f_Age[k]++;
				break;
			}
		}
		//-- if input didn't match with any existing model, initialize the last model with the new input
		if (exist == false) {
			f_Var[f_Num-1] = INIT_FG_VAR;
			f_Mean[f_Num-1] = input;
			f_Age[f_Num-1] = 0;
		}
	}

	//-- instead of previous else, we can force the new input in the most similar model
	//else {
	//	double max_p = -INT_MAX * 1.0;
	//	int max_s = 0;
	//	double max_sig_v = 0.0;
	//	for (int s = 0; s < f_Num; s++) {
	//		double p_v = -(f_Mean[s] - input) * (f_Mean[s] - input) / f_Var[s];
	//		double sig_v = f_Var[s];

	//		//cout << "s:  " << s << "  p_v:  " << p_v << "  log(max_sig_v / sig_v)   " << log(max_sig_v / sig_v) << endl;
	//		if ((max_sig_v / sig_v) > 1 + max_p - p_v + (max_p - p_v) * (max_p - p_v) / 2 + (max_p - p_v) * (max_p - p_v) * (max_p - p_v) / 6 || s == 0) {
	//			max_p = p_v;
	//			max_s = s;
	//			max_sig_v = sig_v;
	//		}
	//	}

	//	float var_temp = (f_Age[max_s] * f_Var[max_s] + (f_Mean[max_s] - (input))) * (f_Mean[max_s] - (input)) / (f_Age[max_s] + 1);
	//	f_Var[max_s] = var_temp < MAX_FG_VAR ? MAX_FG_VAR : (var_temp > MAX_FG_VAR ? MAX_FG_VAR : var_temp);
	//	f_Mean[max_s] = (f_Age[max_s] * f_Mean[max_s] + (input)) / (f_Age[max_s] + 1);
	//	f_Age[max_s]++;
	//}



	//-- update weights
	for (int k = 0; k < NUM_MODELS_FG; k++) {
		totalWeight += f_Age[k];
	}
	if (totalWeight >= INT_MAX - 1) {
		totalWeight /= 2;
		for (int k = 0; k < NUM_MODELS_FG; k++) {
			f_Age[k] /= 2;
			if (f_Age[k] == 0) {
				f_Age[k]++;
				totalWeight++;
			}
			f_Weight[k] = f_Age[k] / totalWeight;
		}
	}
	else {
		for (int k = 0; k < NUM_MODELS_FG; k++) {
			f_Weight[k] = f_Age[k] / totalWeight;
		}
	}

	//-- adjust position
	for (int l = NUM_MODELS_FG - 1; l > 0; l--) {
		if (f_Weight[l] > f_Weight[(l - 1)]) {
			float tempMean = f_Mean[l];
			float tempVar = f_Var[l];
			float tempAge = f_Age[l];
			float tempWeight = f_Weight[l];

			f_Mean[l] = f_Mean[l - 1];
			f_Var[l] = f_Var[l - 1];
			f_Age[l] = f_Age[l - 1];
			f_Weight[l] = f_Weight[l - 1];

			f_Mean[l - 1] = tempMean;
			f_Var[l - 1] = tempVar;
			f_Age[l - 1] = tempAge;
			f_Weight[l - 1] = tempWeight;
		}
	}
}


//============================
//	draw charts
//============================
void DCFG::draw_1d_gaussian(cv::Mat img, float mean, float stddev, cv::Scalar color = cv::Scalar(0, 0, 0)) {
	float* prob = new float[img.cols];
	float frac = 1 / (stddev * sqrt(2 * CV_PI));

	float min_prob = FLT_MAX;
	float max_prob = FLT_MIN;

	mean += 250;

	for (int x = 0; x < img.cols; x++) {
		prob[x] = frac * exp(-0.5 * (x - mean) * (x - mean) / (stddev * stddev));

		if (prob[x] > max_prob)
			max_prob = prob[x];

		if (prob[x] < min_prob)
			min_prob = prob[x];
	}

	float prev = 0;
	for (int x = 0; x < img.cols; x++) {
		float p = 30 + 100 * (1 - (prob[x] / (max_prob - min_prob)));

		cv::line(img, cv::Point(x - 1, prev), cv::Point(x, p), color);
		prev = p;
	}

	cv::line(img, cv::Point(mean, 30), cv::Point(mean, 130), cv::Scalar(0));
	cv::line(img, cv::Point(mean - stddev, 30), cv::Point(mean - stddev, 130), cv::Scalar(128));
	cv::line(img, cv::Point(mean + stddev, 30), cv::Point(mean + stddev, 130), cv::Scalar(128));
}


void DCFG::draw_1d_draw_mixture(cv::Mat img, float* data, float* weights, int count_per_component, int num_components) {
	std::vector<cv::Scalar> palette;
	palette.push_back(cv::Scalar(0, 0, 255));   // red
	palette.push_back(cv::Scalar(255, 0, 0));
	palette.push_back(cv::Scalar(0, 255, 0));
	palette.push_back(cv::Scalar(0, 0, 0));
	palette.push_back(cv::Scalar(192, 192, 192));

	int total = count_per_component * num_components;
	for (int i = 0; i < total; i++) {
		float* wt = &weights[i * num_components];

		cv::Scalar color = cv::Scalar(0);
		for (int j = 0; j < num_components; j++) {
			color += wt[j] * palette[j];
		}

		float pt = data[i];
		cv::line(img, cv::Point(pt, 10), cv::Point(pt, 20), color);
	}
}


void DCFG::draw_1d_gaussian_mixture(cv::Mat img, float* mean, float* stddev, float* pi, int num_components) {
	float* prob = new float[img.cols] {};
	float* frac = new float[num_components] {};

	for (int j = 0; j < num_components; j++) {
		frac[j] = 1.0 / (stddev[j] * sqrt(2 * CV_PI));
	}

	float min_prob = FLT_MAX;
	float max_prob = FLT_MIN;

	for (int x = 0; x < img.cols; x++) {
		for (int j = 0; j < num_components; j++) {
			prob[x] = pi[j] * frac[j] * exp(-0.5 * (x - mean[j]) * (x - mean[j]) / (stddev[j] * stddev[j]));
		}

		if (prob[x] > max_prob)
			max_prob = prob[x];

		if (prob[x] < min_prob)
			min_prob = prob[x];
	}

	float prev = 0;
	for (int x = 0; x < img.cols; x++) {
		float p = 30 + 100 * (1 - (prob[x] / (max_prob - min_prob)));
		cv::line(img, cv::Point(x - 1, prev), cv::Point(x, p), cv::Scalar(0));
		prev = p;
	}

	for (int i = 0; i < num_components; i++) {
		cv::line(img, cv::Point(mean[i], 30), cv::Point(mean[i], 130), cv::Scalar(0));
		//cv::line(img, cv::Point(mean[i] - stddev[i], 30), cv::Point(mean[i] - stddev[i], 130), cv::Scalar(128));
		//cv::line(img, cv::Point(mean[i] + stddev[i], 30), cv::Point(mean[i] + stddev[i], 130), cv::Scalar(128));
	}
}



//============================
//	ViBe
//============================
const unsigned char DCFG::BACK_GROUND = 0;
const unsigned char DCFG::FORE_GROUND = 255;

//The first method: the most primitive vibe grayscale channel
void DCFG::originalVibe_Init_GRAY(const cv::Mat& firstFrame)
{
	int height = firstFrame.rows;
	int width = firstFrame.cols;
	//The background model allocates memory
	backgroundModel.clear();
	for (int index = 0; index < this->numberSamples; index++)
	{
		backgroundModel.push_back(cv::Mat::zeros(height, width, CV_8UC1));
	}
	// Random number
	cv::RNG rng;
	int cshift;
	int rshift;
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			if (c < neighborWidth / 2 || c > width - neighborWidth / 2 - 1 || r < neighborHeight / 2 || r > height - neighborHeight / 2 - 1)
			{
				/* There are many ways to generate random numbers */
				/*
				cshift = randu<int>()%neighborWidth - neighborWidth/2;
				rshift = randu<int>()%neighborHeight - neighborHeight/2;
				*/
				cshift = rand() % neighborWidth - neighborWidth / 2;
				rshift = rand() % neighborHeight - neighborHeight / 2;

				for (std::vector<cv::Mat>::iterator it = backgroundModel.begin(); it != backgroundModel.end(); it++)
				{
					for (;;)
					{
						/*
						cshift = rng.uniform(-neighborWidth/2,neighborWidth/2 + 1);
						rshift = rng.uniform(-neighborHeight/2,neighborHeight/2 +1 );
						*/
						cshift = abs(cv::randu<int>() % neighborWidth) - neighborWidth / 2;
						rshift = abs(cv::randu<int>() % neighborHeight) - neighborHeight / 2;

						if (!(cshift == 0 && rshift == 0))
							break;
					}
					if (c + cshift < 0 || c + cshift >= width)
						cshift *= -1;
					if (r + rshift < 0 || r + rshift >= height)
						rshift *= -1;
					(*it).at<uchar>(r, c) = firstFrame.at<uchar>(r + rshift, c + cshift);
				}
			}
			else
			{
				for (std::vector<cv::Mat>::iterator it = backgroundModel.begin(); it != backgroundModel.end(); it++)
				{
					for (;;)
					{
						/*
						cshift = rng.uniform(-neighborWidth/2,neighborWidth/2 + 1);
						rshift = rng.uniform(-neighborHeight/2,neighborHeight/2 +1 );
						*/
						cshift = abs(cv::randu<int>() % neighborWidth) - neighborWidth / 2;
						rshift = abs(cv::randu<int>() % neighborHeight) - neighborHeight / 2;
						if (!(cshift == 0 && rshift == 0))
							break;
					}
					(*it).at<uchar>(r, c) = firstFrame.at<uchar>(r + rshift, c + cshift);
				}
			}
		}
	}
}


void DCFG::originalVibe_ClassifyAndUpdate_GRAY(const cv::Mat& frame, cv::OutputArray& _segmentation)
{
	int width = frame.cols;
	int height = frame.rows;
	int rshift;
	int cshift;
	_segmentation.create(frame.size(), CV_8UC1);
	cv::Mat segmentation = _segmentation.getMat();

	cv::RNG rng;
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			int count = 0;
			unsigned char pixel = frame.at<uchar>(r, c);
			// Let the pixel compare with the backgroundModel in the background template
			for (std::vector<cv::Mat>::iterator it = backgroundModel.begin(); it != backgroundModel.end(); it++)
			{
				if (abs(int(pixel) - int((*it).at<uchar>(r, c))) < distanceThreshold)
				{
					count++;
					// Loop to a certain stage, judge whether the value of count is greater than minMatch, update the background model
					if (count >= this->minMatch)
					{
						int random = rng.uniform(0, this->updateFactor);
						if (random == 0)
						{
							int updateIndex = rng.uniform(0, this->numberSamples);
							backgroundModel[updateIndex].at<uchar>(r, c) = pixel;
						}
						random = rng.uniform(0, this->updateFactor);
						if (random == 0)
						{
							if (c < neighborWidth / 2 || c > width - neighborWidth / 2 - 1 || r < neighborHeight / 2 || r > height - neighborHeight / 2 - 1)
							{
								for (;;)
								{
									/*
									 cshift = rng.uniform(-neighborWidth/2,neighborWidth/2 + 1);
									 rshift = rng.uniform(-neighborHeight/2,neighborHeight/2 +1 );
									*/
									cshift = abs(cv::randu<int>() % neighborWidth) - neighborWidth / 2;
									rshift = abs(cv::randu<int>() % neighborHeight) - neighborHeight / 2;
									if (!(cshift == 0 && rshift == 0))
										break;
								}
								if (c + cshift < 0 || c + cshift >= width)
									cshift *= -1;
								if (r + rshift < 0 || r + rshift >= height)
									rshift *= -1;
								int updateIndex = rng.uniform(0, this->numberSamples);
								backgroundModel[updateIndex].at<uchar>(r + rshift, c + cshift) = pixel;
							}
							else
							{
								for (;;)
								{
									/*
									cshift = rng.uniform(-neighborWidth/2,neighborWidth/2 + 1);
									rshift = rng.uniform(-neighborHeight/2,neighborHeight/2 +1 );
									*/
									cshift = abs(cv::randu<int>() % neighborWidth) - neighborWidth / 2;
									rshift = abs(cv::randu<int>() % neighborHeight) - neighborHeight / 2;
									if (!(cshift == 0 && rshift == 0))
										break;
								}
								int updateIndex = rng.uniform(0, this->numberSamples);
								backgroundModel[updateIndex].at<uchar>(r + rshift, c + cshift) = pixel;
							}
						}
						segmentation.at<uchar>(r, c) = this->BACK_GROUND;
						break;
					}
				}
			}
			if (count < this->minMatch)
				segmentation.at<uchar>(r, c) = this->FORE_GROUND;
		}
	}
}


//The third method: BGR channel
void DCFG::originalVibe_Init_BGR(const cv::Mat& fristFrame)
{
	int height = fristFrame.rows;
	int width = fristFrame.cols;
	//The background model allocates memory
	backgroundModel.clear();
	for (int index = 0; index < this->numberSamples; index++)
	{
		backgroundModel.push_back(cv::Mat::zeros(height, width, CV_8UC3));
	}
	// Random number
	cv::RNG rng;
	int cshift;
	int rshift;
	for (int r = 0; r < height; r++)
	{
		for (int c = 0; c < width; c++)
		{
			if (c < neighborWidth / 2 || c > width - neighborWidth / 2 - 1 || r < neighborHeight / 2 || r > height - neighborHeight / 2 - 1)
			{
				/*
				 Initialize the background model: start
				*/
				for (std::vector<cv::Mat>::iterator iter = backgroundModel.begin(); iter != backgroundModel.end(); iter++)
				{
					for (;;)
					{
						cshift = abs(cv::randu<int>() % neighborWidth) - neighborWidth / 2;
						rshift = abs(cv::randu<int>() % neighborHeight) - neighborHeight / 2;
						if (!(cshift == 0 && rshift == 0))
							break;
					}
					if (c + cshift < 0 || c + cshift >= width)
						cshift *= -1;
					if (r + rshift < 0 || r + rshift >= height)
						rshift *= -1;
					(*iter).at<cv::Vec3b>(r, c) = fristFrame.at<cv::Vec3b>(r + rshift, c + cshift);
				}
			}
			/* Initialize the background model: end */
			else
			{
				/* ******Initialize the background model: start ***** */
				for (std::vector<cv::Mat>::iterator iter = backgroundModel.begin(); iter != backgroundModel.end(); iter++)
				{
					for (;;)
					{
						cshift = abs(cv::randu<int>() % neighborWidth) - neighborWidth / 2;
						rshift = abs(cv::randu<int>() % neighborHeight) - neighborHeight / 2;
						if (!(cshift == 0 && rshift == 0))
							break;
					}
					(*iter).at<cv::Vec3b>(r, c) = fristFrame.at<cv::Vec3b>(r + rshift, c + cshift);
				}
				/* ****Initialize the background model: end***** */
			}
		}
	}
}

float DCFG::distanceL2(const cv::Vec3b& src1, const  cv::Vec3b& src2)
{
	return pow(pow(src1[0] - src2[0], 2.0) + pow(src1[1] - src2[1], 2.0) + pow(src1[2] - src2[2], 2.0), 0.5);
}
int DCFG::distanceL1(const cv::Vec3b& src1, const  cv::Vec3b& src2)
{
	return abs(src1[0] - src2[0]) + abs(src1[1] - src2[1]) + abs(src1[2] - src2[2]);
}

void DCFG::originalVibe_ClassifyAndUpdate_BGR(const cv::Mat& frame, cv::OutputArray& _segmentation)
{ // *Number 1
	int height = frame.rows;
	int width = frame.cols;
	int cshift;
	int rshift;
	_segmentation.create(frame.size(), CV_8UC1);
	cv::Mat segmentation = _segmentation.getMat();

	cv::RNG rng;

	for (int r = 0; r < height; r++)
	{ // Number 1-1
		for (int c = 0; c < width; c++)
		{ // Number 1-1-1
			int count = 0;
			cv::Vec3b pixel = frame.at<cv::Vec3b>(r, c);
			for (std::vector<cv::Mat>::iterator iter = backgroundModel.begin(); iter != backgroundModel.end(); iter++)
			{ // Number 1-1-1-1
				//
				//
				if (distanceL1(pixel, (*iter).at<cv::Vec3b>(r, c)) < 4.5 * this->distanceThreshold)
				{
					count++;
					if (count >= this->minMatch)
					{
						//The first step: update the model update
						/* *********Start to update the model************ */
						int random = rng.uniform(0, this->updateFactor);
						if (random == 0)
						{
							int updateIndex = rng.uniform(0, this->numberSamples);
							backgroundModel[updateIndex].at<cv::Vec3b>(r, c) = pixel;
						}

						random = rng.uniform(0, this->updateFactor);
						if (random == 0)
						{
							/****************************************/
							if (c < neighborWidth / 2 || c > width - neighborWidth / 2 - 1 || r < neighborHeight / 2 || r > height - neighborHeight / 2 - 1)
							{
								for (;;)
								{
									cshift = abs(cv::randu<int>() % neighborWidth) - neighborWidth / 2;
									rshift = abs(cv::randu<int>() % neighborHeight) - neighborHeight / 2;
									if (!(cshift == 0 && rshift == 0))
										break;
								}
								if (c + cshift < 0 || c + cshift >= width)
									cshift *= -1;
								if (r + rshift < 0 || r + rshift >= height)
									rshift *= -1;
								int updateIndex = rng.uniform(0, this->numberSamples);
								backgroundModel[updateIndex].at<cv::Vec3b>(r + rshift, c + cshift) = pixel;
							}
							else
							{
								for (;;)
								{
									cshift = abs(rand() % neighborWidth) - neighborWidth / 2;
									rshift = abs(rand() % neighborHeight) - neighborHeight / 2;
									if (!(cshift == 0 && rshift == 0))
										break;
								}
								int updateIndex = rng.uniform(0, this->numberSamples);
								backgroundModel[updateIndex].at<cv::Vec3b>(r + rshift, c + cshift) = pixel;
							}
							/****************************************/
						}
						/*
						*********End update model************
						*/
						//The second step: classify
						segmentation.at<uchar>(r, c) = this->BACK_GROUND;
						break;
					}
				}
			} // Number 1-1-1-1
			if (count < this->minMatch)//classify
				segmentation.at<uchar>(r, c) = this->FORE_GROUND;
		} // Number 1-1-1
	} // Number 1-1

} // *Number 1