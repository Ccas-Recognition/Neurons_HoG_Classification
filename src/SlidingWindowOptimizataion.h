#ifndef SLIDING_WINODOW_OPTIMIZATION
#define SLIDING_WINODOW_OPTIMIZATION

#include "classifier.h"
#include "SlidingWindow.h"
#include "ImageRecognition.h"
#include <string>

using namespace std;
using HOGFeatureClassifier::TModel;
using ImageRecognition::SlidingRect;

namespace ImageRecognition
{
	void GetRectsFromImage(vector<SlidingRect> &rects, const Mat &image, const TModel &model, RecognitionStatistics &stat);
	float FindOptimalThresholdForModel(const string &images_list, const TModel &model, RecognitionStatistics &stat);
}
#endif