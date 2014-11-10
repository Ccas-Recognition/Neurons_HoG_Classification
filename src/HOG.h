
#ifndef HOG_FEATURES_AND_CLASSIFIER_MMP
#define HOG_FEATURES_AND_CLASSIFIER_MMP

#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "linear.h"


//#include <opencv2/core/core.hpp>
#include "opencv2/opencv.hpp"

namespace HOGFeatureClassifier
{
	//Context {
	const vector<int> blockSizeX = { 4, 6 };
	const vector<int> blockSizeY = { 4, 6 };
	const int dirSegSize = 8;

	const int nonlinear_n = 4;//2
	const float nonlinear_L = 0.5;
	const double param_C = 0.08;
	//}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

	using namespace cv;
using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list);

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
	const TLabels& labels,
	const string& prediction_file);

// Counting module and direction of gradients (3.3)
Mat countModAndDirOfGrad(const Mat &in);

vector<float> HOG(const int blockSizeX, const int blockSizeY, const int dirSegSize, const Mat &image);

// Exatract features from dataset.
void ExtractFeatures(const TFileList& file_list, TFeatures* features);

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file);

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
	const string& model_file,
	const string& prediction_file);

}//HOGFeatureClassifier
#endif //HOG_FEATURES_AND_CLASSIFIER_MMP