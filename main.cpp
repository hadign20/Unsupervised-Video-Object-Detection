#include "opencv2/opencv.hpp"
#include "opencv2/bgsegm.hpp"
#include <fstream>
#include <string>
#include <time.h> 
#include "DCFG.h"

using namespace cv;
using namespace std;

#define DEBUG 0

string getFileName(string filePath);
string getFolderName(string filePath);
string getTimeString(int numFrame, int frameRate);
void setLabel(cv::Mat& im, const std::string label, const cv::Point& or , const cv::Scalar col, const double& fontScale);
void loadConfig(char* ch);


// Default Parameters (can be changed in config.xml)
int skipFrames = 0;
bool scale = false;
int scaleRate = 2;
int preframes = 100;

//video
int main(int argc, char **argv) {
	std::clock_t t1, t2;

	loadConfig("./config.xml");

	Mat frame, bg, mask;

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

	// Initialization
	VideoCapture video(videoName);
	if (!video.isOpened()){
		throw "ERROR: video cannot be opened";
	}

	int frameRate = video.get(CAP_PROP_FPS);

	//-- set results and config paths
	string vname, folder;
	vname = getFileName(videoName);
	folder = getFolderName(videoName);

	video >> frame;
	if (scale)
		cv::resize(frame, frame, cv::Size(), 1.0 / scaleRate, 1.0 / scaleRate);

	DCFG* dcfg = new DCFG(frame);

	std::cout << "Processing " << vname << " ..." << endl;

	//-- dcfg videos
	VideoWriter w1("./results/" + vname + "_" + "fgAndCandid.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter w2("./results/" + vname + "_" + "fgMask.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter w3("./results/" + vname + "_" + "heatMap.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), true);
	VideoWriter w4("./results/" + vname + "_" + "debug.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), true);
	VideoWriter w5("./results/" + vname + "_" + "flowField.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), true);
	VideoWriter w6("./results/" + vname + "_" + "fgConfMap.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter w7("./results/" + vname + "_" + "watershed.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), true);
	VideoWriter w8("./results/" + vname + "_" + "klt.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), true);
	VideoWriter w9("./results/" + vname + "_" + "mask.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), false);
	VideoWriter w10("./results/" + vname + "_" + "bg.avi", CV_FOURCC('M', 'P', '4', '2'), frameRate, frame.size(), true);


	//-- skip frames
	for (int i = 0; i < skipFrames; i++)
		video >> frame;

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
		dcfg->apply(frame, fgMaskDCFG, bg);
		//pMOG->apply(frame, fgMaskMOG);
		//pMOG2->apply(frame, fgMaskMOG2);
		//pGMG->apply(frame, fgMaskGMG);
		//pKNN->apply(frame, fgMaskKNN);
		//pCNT->apply(frame, fgMaskCNT);
		//pGSOC->apply(frame, fgMaskGSOC);
		//pLSBP->apply(frame, fgMaskLSBP);


		cv::medianBlur(fgMaskDCFG, fgMaskDCFG, 5);
		//cv::medianBlur(fgMaskMOG2, fgMaskMOG2, 5);
		//cv::medianBlur(fgMaskGMG, fgMaskGMG, 5);
		//cv::medianBlur(fgMaskKNN, fgMaskKNN, 5);
		//cv::medianBlur(fgMaskMOG, fgMaskMOG, 5);
		//cv::medianBlur(fgMaskCNT, fgMaskCNT, 5);
		//cv::medianBlur(fgMaskGSOC, fgMaskGSOC, 5);
		//cv::medianBlur(fgMaskLSBP, fgMaskLSBP, 5);

		
		//-- choose the foreground mask
		fgMaskDCFG.copyTo(mask);
		//fgMaskMOG2.copyTo(mask);
		//fgMaskGMG.copyTo(mask);
		//fgMaskKNN.copyTo(mask);
		//fgMaskMOG.copyTo(mask);
		//fgMaskCNT.copyTo(mask);
		//fgMaskGSOC.copyTo(mask);
		//fgMaskLSBP.copyTo(mask);

		//-- choose the background image (by default it is DCFG)
		//pMOG->getBackgroundImage(bg);
		//pMOG2->getBackgroundImage(bg);
		//pGMG->getBackgroundImage(bg);
		//pKNN->getBackgroundImage(bg);
		//pCNT->getBackgroundImage(bg);
		//pGSOC->getBackgroundImage(bg);
		//pLSBP->getBackgroundImage(bg);
		//--------------------------------------------------


#if DEBUG

		//imshow("MOG", fgMaskMOG);
		//imshow("MOG2", fgMaskMOG2);
		//imshow("GMG", fgMaskGMG);
		//imshow("KNN", fgMaskKNN);
		//imshow("CNT", fgMaskCNT);
		//imshow("GSOC", fgMaskGSOC);
		//imshow("LSBP", fgMaskLSBP);
		//imshow("GFM", fgMaskGFM);
		//imshow("DCFG", fgMaskDCFG);

		//imshow("frame", frame);
		//imshow("bg", bg);
		//imshow("mask", mask);

		//-- dcfg videos
		w1 << dcfg->fgAndCandid;
		w2 << dcfg->finalFG;
		w2 << fgMaskDCFG;
		w3 << dcfg->heatMap;
		w4 << dcfg->debugImg;
		w5 << dcfg->flowField;
		w6 << dcfg->fgConfMap;
		w7 << dcfg->watershedImage;
		w8 << dcfg->testImg;
		w9 << mask;
		w10 << bg;

#endif
		

		key = cvWaitKey(1);
	}

	t2 = std::clock();




	//-- dcfg videos
	w1.release();
	w2.release();
	w3.release();
	w4.release();
	w5.release();
	w6.release();
	w7.release();
	w8.release();
	w9.release();
	w10.release();


	
	float t_diff((float)t2 - (float)t1);
	float seconds = t_diff / CLOCKS_PER_SEC;
	float mins = seconds / 60.0;
	float hrs = mins / 60.0;
	cout << "\nExecution Time (mins): " << mins << "\n";
	cout << "Execution Time (secs): " << seconds << "\n";
	cout << "\navgTime: " << dcfg->avgTime << endl;
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

std::string getFileName(std::string filePath)
{
	std::string rawname, seperator;
	if (filePath.find_last_of('\\') == string::npos)
		seperator = '/';
	else
		seperator = '\\';

	std::size_t dotPos = filePath.rfind('.');
	std::size_t sepPos = filePath.rfind(seperator);

	rawname = filePath.substr(sepPos + 1, dotPos - sepPos - 1);

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
	scale = cvReadIntByName(fs, 0, "isScale", 0);
	scaleRate = cvReadIntByName(fs, 0, "scaleRate", 2);

	cvReleaseFileStorage(&fs);
}