#include <vector>
#include <opencv2/core/core.hpp>


#include "HOG.h"
#include "HOG_Functor.h"

namespace HOGFeatureClassifier
{
	using namespace std;
	using namespace cv;

	bool HoGResponseFunctor::InitModel(const TModel *_model)
	{
		model = auto_ptr< struct TModel >(new TModel(*_model));
		return true;
	}

	bool HoGResponseFunctor::InitModel(string model_name)
	{
		model = auto_ptr< struct TModel >(new TModel());

		bool loaded = model->Load(model_name);
		if (!loaded)
			return false;
		return true;
	}
	/*
	Mat ComputeintegralImage( const Mat &input_image )
	{
		Mat intMod(input_image.rows, input_image.cols, CV_64FC1);
		
		intMod.at<double>(0, 0) = double(input_image.at<Vec2f>(0, 0).val[0]);

		for (int y = 1; y < input_image.rows; ++y)
		{
			int x = 0;
			intMod.at<double>(y, x) = intMod.at<double>(y-1, x) + double(input_image.at<Vec2f>(y, x).val[0]);
		}
		for (int x = 1; x < input_image.cols; ++x)
		{
			int y = 0;
			intMod.at<double>(y, x) = intMod.at<double>(y, x - 1) + double(input_image.at<Vec2f>(y, x).val[0]);
		}
		for (int y = 1; y < input_image.rows; ++y)
		{
			for (int x = 1; x < input_image.cols; ++x)
			{
				intMod.at<double>(y, x) = double(input_image.at<Vec2f>(y, x).val[0])
					+ intMod.at<double>(y - 1, x) 
					+ intMod.at<double>(y, x - 1) 
					- intMod.at<double>(y - 1, x - 1);
			}
		}
		return intMod;
	}
	*/
	Mat ComputeintegralImage(const Mat &input_image)
	{
		Mat intMod(input_image.rows, input_image.cols, CV_64FC1);

		intMod.at<double>(0, 0) = double(input_image.at<uchar>(0, 0));

		for (int y = 1; y < input_image.rows; ++y)
		{
			int x = 0;
			intMod.at<double>(y, x) = intMod.at<double>(y - 1, x) + double(input_image.at<uchar>(y, x));
		}
		for (int x = 1; x < input_image.cols; ++x)
		{
			int y = 0;
			intMod.at<double>(y, x) = intMod.at<double>(y, x - 1) + double(input_image.at<uchar>(y, x));
		}
		for (int y = 1; y < input_image.rows; ++y)
		{
			for (int x = 1; x < input_image.cols; ++x)
			{
				intMod.at<double>(y, x) = double(input_image.at<uchar>(y, x))
					+ intMod.at<double>(y - 1, x)
					+ intMod.at<double>(y, x - 1)
					- intMod.at<double>(y - 1, x - 1);
			}
		}

		#if 0
		cout << endl;
		for (int y = 0; y < 10; ++y)
		{
			for (int x = 0; x < 10; ++x)
				cout << intMod.at<double>(y, x) << " ";
			cout << endl;
		}
		#endif
		return intMod;
	}
	bool HoGResponseFunctor::Init(const Mat& _image, const Mat& _additionalImage)
	{
		DEBUG_COUNT_OPERATOR = 0;
		image = _image;
		modDir = countModAndDirOfGrad(image);
		integralImage = ComputeintegralImage(_additionalImage);
		return true;
	}

	float HoGResponseFunctor::ComputeFastPredict(int x, int y, int w, int h)
	{
		double sum = (
			integralImage.at<double>(y + h - 2, x + w - 2)
			- integralImage.at<double>(y, x + w - 2)
			- integralImage.at<double>(y + h - 2, x)
			+ integralImage.at<double>(y, x));
		sum /= (h - 2)*(w - 2);
		return float(sum);
	}

	float HoGResponseFunctor::operator()(int pos_x, int pos_y, int width, int height)
	{
		float fastPredict = ComputeFastPredict(pos_x, pos_y, width, height);
		if (fastPredict < 0.05 || fastPredict > 0.8)
		{
			#if 0
				++DEBUG_COUNT_OPERATOR;
			#endif
			return 0.0f;
		}

		const Mat part = modDir(Range(pos_y, pos_y + height), Range(pos_x, pos_x + width));
        ExtractFeaturesForSample(part, features);
		TClassifier::ConvertFeaturesToClassifierType(features, classifier_features);
		float value = float(TClassifier::ComputePredictValue(classifier_features, *(model.get())));
		return value;
	}
}
