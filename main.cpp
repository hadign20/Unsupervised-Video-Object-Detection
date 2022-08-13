#include <opencv2/opencv.hpp>
#include "opencv2/bgsegm.hpp"
#include <fstream>
#include <string>
#include <time.h> 

#include "GFM.h"
#include "DCFG.h"
#include "package_tracking/BlobTracking.h"
#include "ChromacityShadRem.h"
#include "GeometryShadRem.h"
#include "LrTextureShadRem.h"
#include "PhysicalShadRem.h"
#include "SrTextureShadRem.h"
#include "MultilayerShadRem.h"
#include "MCSS.h"
#include "Shadow.h"
#include "Shadow2.h"
#include "Speed.h"
#include "MovingShadRem.h"

using namespace cv;
using namespace std;

#define comparison_debug 0
#define run_debug 0

string getFileName(string filePath, int starttime, int endtime);
string getFolderName(string filePath);
string getTimeString(int numFrame, int frameRate);
void setLabel(cv::Mat& im, const std::string label, const cv::Point& or , const cv::Scalar col, const double& fontScale);
void loadConfig(char* ch);


//Parameters
bool detect_con, detect_slow, detect_stop, detect_wrongw, detect_pe;
int skipFrames = 0;
int frameRate = 15;
bool scale = false;
int scaleRate = 2;
int preframes = 100;
int starttime = 0;
int endtime = 0;



//video
int main(int argc, char **argv){

	std::clock_t t1, t2, t1s, t2s;
	double shadow_time_sum = 0.0;

	loadConfig("./Src/config/config.xml");

	Mat frame, input, diff, bg, mask;

	//-- different shadow detections
	cv::Mat chrMask, phyMask, geoMask, srTexMask, lrTexMask, mulTexMask, fgrMask, shadMask, shad2Mask, msrMask, hsvFrame;

	//-- different background subtractions
	Mat fgMaskMOG, fgMaskMOG2, fgMaskGMG, fgMaskKNN, fgMaskGSOC, fgMaskLSBP, fgMaskCNT, fgMaskGFM, fgMaskDCFG;
	Ptr< BackgroundSubtractor> pMOG = cv::bgsegm::createBackgroundSubtractorMOG();
	Ptr< BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(500,16,false);
	Ptr< BackgroundSubtractor> pGMG = cv::bgsegm::createBackgroundSubtractorGMG();
	Ptr< BackgroundSubtractor> pKNN = createBackgroundSubtractorKNN(500,400,false);
	Ptr<BackgroundSubtractor> pCNT = cv::bgsegm::createBackgroundSubtractorCNT();
	Ptr<BackgroundSubtractor> pGSOC = cv::bgsegm::createBackgroundSubtractorGSOC();
	Ptr<BackgroundSubtractor> pLSBP = cv::bgsegm::createBackgroundSubtractorLSBP();

	int numFrame = skipFrames, key = 0;
	string videoName = argv[1];
	string initialTime[2];

	GFM* gfm = new GFM;
	Ptr<BackgroundSubtractorMOG2> opencv = createBackgroundSubtractorMOG2(500, 5, false);

	ChromacityShadRem chr;
	PhysicalShadRem phy;
	GeometryShadRem geo;
	SrTextureShadRem srTex;
	LrTextureShadRem lrTex;
	MultilayerShadRem mulTex;
	MCSS fgr;
	MCSS_Param p;

	// Initialization
	VideoCapture video(videoName);
	VideoCapture pretrain(videoName);
	if (!video.isOpened()){
		throw "ERROR: video cannot be opened";
	}

	//-- set shadow methods parameters
	p = fgr.getParameters();
	p.alpha = 0.000621;
	p.threshold1 = 1.7;
	p.threshold2 = 12.5;
	p.isMGThrFixed = true;
	p.hasFringe = true;
	p.mgThrFixed = Vec3f(0.23, 0.23, 0.23);
	fgr.setParameters(p);

	//-- set results and config paths
	string vname, folder;
	vname = getFileName(videoName, starttime, endtime);
	folder = getFolderName(videoName);

	video >> frame;
	if (scale)
		cv::resize(frame, frame, cv::Size(), 1.0 / scaleRate, 1.0 / scaleRate);

	Shadow* shadow = new Shadow(frame.size());
	Shadow2* shadow2 = new Shadow2(frame.size());
	MovingShadRem* msr = new MovingShadRem(frame);
	DCFG* dcfg = new DCFG(frame);
	

	std::cout << vname << endl;

	VideoWriter write1("./results/" + vname + "_" + "origblock.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), true);
	VideoWriter write2("./results/" + vname + "_" + "phy.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write3("./results/" + vname + "_" + "geo.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write4("./results/" + vname + "_" + "sr.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write5("./results/" + vname + "_" + "lr.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write6("./results/" + vname + "_" + "l1.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write7("./results/" + vname + "_" + "l2.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write8("./results/" + vname + "_" + "l3.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write18("./results/" + vname + "_" + "l4.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write9("./results/" + vname + "_" + "combine.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write10("./results/" + vname + "_" + "mask.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write11("./results/" + vname + "_" + "sh.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write12("./results/" + vname + "_" + "sh_red.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), true);
	VideoWriter write13("./results/" + vname + "_" + "chr.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write14("./results/" + vname + "_" + "candidate.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write15("./results/" + vname + "_" + "segments.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), true);
	VideoWriter write16("./results/" + vname + "_" + "bgShadow.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write17("./results/" + vname + "_" + "cluster.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), true);
	VideoWriter write19("./results/" + vname + "_" + "final.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write20("./results/" + vname + "_" + "ratio.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), true);
	VideoWriter write21("./results/" + vname + "_" + "old_ratio.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), true);
	VideoWriter write22("./results/" + vname + "_" + "fgr.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write23("./results/" + vname + "_" + "fgr_lumRatio2.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), true);

	//VideoWriter write24("./results/" + vname + "_" + "debug.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, cv::Size(frame.cols * 2, frame.rows), false);
	VideoWriter write24("./results/" + vname + "_" + "debug.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), true);
	VideoWriter write26("./results/" + vname + "_" + "old_mask.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write27("./results/" + vname + "_" + "new_mask.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter write25("./results/" + vname + "_" + "small.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, cv::Size(frame.cols / 4, frame.rows / 4), true);


	input = Mat(frame.rows, frame.cols, CV_32FC(3));
	diff = Mat(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
	gfm->initialization(input);
	gfm->setAlpha(0.004);

	//-- skip frames
	for (int i = 0; i < skipFrames; i++){
		video >> frame;
		pretrain >> frame;
	}

	//-- pretrain frames
	for (int i = 0; i < preframes; i++){
		pretrain >> frame;
		if (frame.empty()){
			break;
		}
		if (scale)
			cv::resize(frame, frame, cv::Size(), 1.0 / scaleRate, 1.0 / scaleRate);
		frame.convertTo(input, CV_32FC(3));
		gfm->process(input, &fgMaskGFM, &bg, &diff);
	}


	//-- start timer
	t1 = clock();

	//-- process
	while (1){

		numFrame++;
		video >> frame;

		if (frame.empty()){
			break;
		}

		if (scale)
			cv::resize(frame, frame, cv::Size(), 1.0 / scaleRate, 1.0 / scaleRate);



		//======================================
		//-- foreground segmentation methods
		//======================================
		//frame.convertTo(input, CV_32FC(3));
		gfm->process(input, &fgMaskGFM, &bg, &diff);
		dcfg->apply(frame, fgMaskDCFG, bg);
		//dcfg->Run();
		//pMOG->apply(frame, fgMaskMOG);
		//pMOG2->apply(frame, fgMaskMOG2);
		//pGMG->apply(frame, fgMaskGMG);
		pKNN->apply(frame, fgMaskKNN);
		//pCNT->apply(frame, fgMaskCNT);
		//pGSOC->apply(frame, fgMaskGSOC);
		//pLSBP->apply(frame, fgMaskLSBP);

		//cv::medianBlur(fgMaskGFM, fgMaskGFM, 5);
		cv::medianBlur(fgMaskDCFG, fgMaskDCFG, 5);
		//cv::medianBlur(fgMaskMOG2, fgMaskMOG2, 5);
		//cv::medianBlur(fgMaskGMG, fgMaskGMG, 5);
		cv::medianBlur(fgMaskKNN, fgMaskKNN, 9);
		//cv::medianBlur(fgMaskMOG, fgMaskMOG, 5);
		//cv::medianBlur(fgMaskCNT, fgMaskCNT, 5);
		//cv::medianBlur(fgMaskGSOC, fgMaskGSOC, 5);
		//cv::medianBlur(fgMaskLSBP, fgMaskLSBP, 5);

		//imshow("MOG", fgMaskMOG);
		//imshow("MOG2", fgMaskMOG2);
		//imshow("GMG", fgMaskGMG);
		//imshow("KNN", fgMaskKNN);
		//imshow("CNT", fgMaskCNT);
		//imshow("GSOC", fgMaskGSOC);
		//imshow("LSBP", fgMaskLSBP);
		//imshow("GFM", fgMaskGFM);
		//imshow("DCFG", fgMaskDCFG);
		
		//-- choose the foreground mask
		fgMaskKNN.copyTo(mask);
		//pKNN->getBackgroundImage(bg);
		//--------------------------------------------------


		//-- timer
		t1s = clock();
		//-------------

		//chr.removeShadows(frame, mask, bg, chrMask);
		//phy.removeShadows(frame, mask, bg, phyMask);
		//geo.removeShadows(frame, mask, bg, geoMask);
		//srTex.removeShadows(frame, mask, bg, srTexMask);
		//lrTex.removeShadows(frame, mask, bg, lrTexMask);
		//mulTex.removeShadows(frame, mask, bg, mulTexMask);
		//shadow->removeShadows(frame, mask, bg, shadMask);
		////shadow2->removeShadows(frame, mask, bg, shad2Mask);
		//fgr(frame, bg, mask, fgrMask);
		//msr->removeShadows(frame, mask, bg, msrMask);
		

		//-- timer
		t2s = clock();
		float t_diff_s((float)t2s - (float)t1s);
		shadow_time_sum += t_diff_s;
		//-------------

		

		
		//-- just representation
		if (run_debug) {
		//	//Mat represent1, represent2;
		//	//Mat difference1, difference2;

		//	//subtract(mask, mulTexMask, difference1);
		//	//subtract(mask, shadMask, difference2);
		//	//frame.copyTo(represent1);
		//	//frame.copyTo(represent2);
		//	//add(represent1, Scalar(0, 0, 250), represent1, difference1);
		//	//add(represent2, Scalar(0, 0, 250), represent2, difference2);
		//	//imshow("represent1", represent1);
		//	//imshow("represent2", represent2);

			Mat phyRep, geoRep, srTexRep, lrTexRep, chrRep, mulTexRep;
			phyMask.copyTo(phyRep);
			geoMask.copyTo(geoRep);
			srTexMask.copyTo(srTexRep);
			lrTexMask.copyTo(lrTexRep);
			chrMask.copyTo(chrRep);
			mulTexMask.copyTo(mulTexRep);
			phyRep.setTo(128, mask - phyMask);
			geoRep.setTo(128, mask - geoMask);
			srTexRep.setTo(128, mask - srTexMask);
			lrTexRep.setTo(128, mask - lrTexMask);
			chrRep.setTo(128, mask - chrMask);
			mulTexRep.setTo(128, mask - mulTexMask);

			imshow("result", fgrMask);

			write1 << bg;
			write2 << phyRep;
			write3 << geoRep;
			write4 << srTexRep;
			write5 << lrTexRep;
			write22 << fgrMask;
			write6 << shadow2->l1Rep;
			write7 << shadow2->l2Rep;
			write8 << shadow2->l3Rep;
			write18 << shadow2->l4Rep;
			write9 << mulTexMask;
			write10 << mask;
			write11 << shadMask;
			//write12 << represent2;
			write13 << chrRep;
			//write14 << shadow->candidShadows;
			write15 << shadow2->sgmentsRep;
			write16 << shadow->bgShadowMask;
			write19 << shadow->finalRep;
			write20 << shadow2->ratioRep;
			//write21 << shadow2->oldRatioRep;
			write23 << fgr.lumRatio2;
		}

		//cv::imshow("debugImg", dcfg->debugImg);
		//write24 << dcfg->debugImg;

		//cv::Mat smallFrame;
		//cv::resize(frame, smallFrame, cv::Size(), 1.0 / 4, 1.0 / 4);

		//write25 << smallFrame;

		write10 << mask;
		write27 << fgMaskDCFG;
		write26 << dcfg->oldMask;

		key = cvWaitKey(1);
	}

	t2 = std::clock();

	//delete shadow;
	//delete msr;

	

	write1.release();
	write2.release();
	write3.release();
	write4.release();
	write5.release();
	write6.release();
	write7.release();
	write8.release();
	write9.release();
	write10.release();
	write11.release();
	write12.release();
	write13.release();
	write14.release();
	write15.release();
	write16.release();
	write17.release();
	write18.release();
	write19.release();
	write20.release();
	write21.release();
	write22.release();
	write23.release();
	write24.release();
	write25.release();
	
	float t_diff((float)t2 - (float)t1);
	float seconds = t_diff / CLOCKS_PER_SEC;
	float mins = seconds / 60.0;
	float hrs = mins / 60.0;
	cout << "\nExecution Time (mins): " << mins << "\n";
	cout << "Execution Time (secs): " << seconds << "\n";

	double shad_ms = (shadow_time_sum / CLOCKS_PER_SEC) * 1000;
	double avg_shad = shad_ms / (double) numFrame;
	cout << "\nAverage shadow time (ms per frame): " << avg_shad << "\n";
	cout << "finish!!" << endl;
	cout << "===================================================" << endl;

	cvWaitKey(0);
	system("Pause");
}


//------------------------------------------------------------------------------------
string getTimeString(int numFrame, int frameRate){
	// Add result to file
	int totalSecs = (numFrame / frameRate);
	int hours = totalSecs / 3600;
	int minutes = (totalSecs % 3600) / 60;
	int seconds = totalSecs % 60;
	string timeString = format("%02d:%02d:%02d", hours, minutes, seconds);
	return timeString;
}

//------------------------------------------------------------------------------------

std::string getFileName(std::string filePath, int starttime, int endtime)
{
	std::string rawname, seperator;
	if (filePath.find_last_of('\\') == string::npos)
		seperator = '/';
	else
		seperator = '\\';

	std::size_t dotPos = filePath.rfind('.');
	std::size_t sepPos = filePath.rfind(seperator);

	rawname = filePath.substr(sepPos + 1, dotPos - sepPos - 1);

	if (starttime != 0 || endtime != 0)
		rawname = rawname + "_" + to_string(starttime) + "-" + to_string(endtime);

	return rawname;
}

std::string getFolderName(std::string filePath)
{
	std::string pathname, foldername, seperator;
	if (filePath.find_last_of('\\') == string::npos)
		seperator = '/';
	else
		seperator = '\\';

	if (filePath.find("data") == string::npos) return "";

	std::size_t sepPos = filePath.rfind(seperator);
	std::size_t dataPos = filePath.find("data");

	if (sepPos - dataPos == 4) return "";

	//-- just the last folder
	//sepPos = pathname.rfind(seperator); // second to last separator
	//foldername = pathname.substr(sepPos + 1);


	//-- all folders after data
	pathname = filePath.substr(dataPos + 4, sepPos - dataPos - 4);

	vector<string> folders;
	std::size_t sepPos2 = 1;
	while (sepPos2 != 0) {
		sepPos2 = pathname.rfind(seperator);
		folders.push_back(pathname.substr(sepPos2 + 1));
		pathname = pathname.substr(0, pathname.size() - sepPos2 - 2);
		if (sepPos2 == pathname.size() - 1)
			pathname = pathname.substr(0, pathname.size() - 1);
	}

	short num_folders = folders.size();
	for (int i = 0; i < num_folders; i++) {
		foldername += folders.back() + "/";
		folders.pop_back();
	}

	return foldername;
}


bool IsPathExist(const std::string& s)
{
	struct stat buffer;
	return (stat(s.c_str(), &buffer) == 0);
}



void setLabel(cv::Mat& im, const std::string label, const cv::Point& or , const cv::Scalar col, const double& fontScale)
{
	int fontface = cv::FONT_HERSHEY_SIMPLEX;
	//int fontface = cv::FONT_HERSHEY_PLAIN;
	int thickness = 2;
	int baseline = 0;

	//cv::Size text = cv::getTextSize(label, fontface, fontScale, thickness, &baseline);
	//cv::rectangle(im, or +cv::Point(0, baseline), or +cv::Point(text.width, -text.height), CV_RGB(0, 0, 0), CV_FILLED);
	cv::putText(im, label, or , fontface, fontScale, cv::Scalar(255, 255, 255), thickness + 1, CV_AA);
	cv::putText(im, label, or , fontface, fontScale, col, thickness, CV_AA);
}


void loadConfig(char* ch){

	CvFileStorage* fs = cvOpenFileStorage(string(ch).c_str(), 0, CV_STORAGE_READ);

	skipFrames = cvReadIntByName(fs, 0, "skipFrames", 0);
	preframes = cvReadIntByName(fs, 0, "preFrames", 150);
	frameRate = cvReadIntByName(fs, 0, "frameRate", 15);
	scale = cvReadIntByName(fs, 0, "isScale", 0);
	scaleRate = cvReadIntByName(fs, 0, "scaleRate", 2);

	cvReleaseFileStorage(&fs);
}