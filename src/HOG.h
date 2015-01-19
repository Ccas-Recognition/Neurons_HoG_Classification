
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

#include <opencv2/core/core.hpp>
#include "consts.h"

#include "SlidingWindow.h"
#include "HOG_Functor.h"

namespace HOGFeatureClassifier
{

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
Mat countModAndDirOfGrad(const Mat &in, const HOGContext& context, RecognitionStatistics &stat);

void HOG(const int blockSizeX, const int blockSizeY, const int dirSegSize, const Mat &modDir, vector<float> &feats, const HOGContext& context, RecognitionStatistics &stat);

void KNearestHOGTest(const string& data_file, const HOGContext& _context, RecognitionStatistics &stat);

void FastPredictForSamples(const TFileList& file_list, vector< float > &fastFeatures, const HOGContext& context, RecognitionStatistics &stat);

float FastPredict(const Mat &modDir, const HOGContext& context, RecognitionStatistics &stat);

void ExtractFeaturesForSample(const Mat& modDir, vector<float> &feats, const HOGContext& context, RecognitionStatistics &stat);

// Exatract features from dataset.
void ExtractFeatures(const TFileList& file_list, TFeatures* features, const HOGContext& context, RecognitionStatistics &stat);

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
TModel TrainClassifier(const string& data_file, const string &images_list, const string& model_file, const HOGContext& context, RecognitionStatistics &stat);

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
	const string& model_file,
	const string& prediction_file,
	RecognitionStatistics &stat);

}//HOGFeatureClassifier
#endif //HOG_FEATURES_AND_CLASSIFIER_MMP