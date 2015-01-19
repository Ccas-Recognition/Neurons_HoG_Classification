#ifndef LAPLACIAN_FEATURES_H
#define LAPLACIAN_FEATURES_H

#include <vector>

#include "ImageRecognition.h"
using HOGFeatureClassifier::HOGContext;

namespace ImageFeatures
{
	float ComputeScaleSpaceLaplacianFeature(int pos, const vector<float> &values, vector<float> &buffer, const HOGContext &context);
}

#endif // LAPLACIAN_FEATURES_H