#include <vector>
#include <opencv2/core/core.hpp>


#include "HOG.h"
#include "HOG_Functor.h"

namespace HOGFeatureClassifier
{
	using namespace std;
	using namespace cv;

	bool HoGResponseFunctor::InitModel(string model_name)
	{
		model = auto_ptr< struct TModel >(new TModel());

		bool loaded = model->Load(model_name);
		if (!loaded)
			return false;
		return true;
	}

	bool HoGResponseFunctor::Init(Mat& _image)
	{
		image = _image;
		modDir = countModAndDirOfGrad(image);
		return true;
	}

	float HoGResponseFunctor::operator()(int pos_x, int pos_y, int width, int height)
	{
		const Mat part = modDir(Range(pos_y, pos_y + height), Range(pos_x, pos_x + width));
        ExtractFeaturesForSample(part, features);
		TClassifier::ConvertFeaturesToClassifierType(features, classifier_features);
		float value = float(TClassifier::ComputePredictValue(classifier_features, *(model.get())));
		return value;
	}
}
