#ifndef IMAGE_RECOGNITION_H
#define IMAGE_RECOGNITION_H

#include<opencv2/core/core.hpp>

#include <vector>
#include <string>
#include <cassert>

using namespace std;
using namespace cv;

//Forward Declaration:
namespace HOGFeatureClassifier { class TModel; }
namespace ImageRecognition { class SlidingRect; }

//Main Interface:
namespace ImageRecognition
{
	HOGFeatureClassifier::TModel TrainHOGClassifier(const string& data_file, const string &images_list, const string& model_file);
	void PredictData(const string& data_file, const string& model_file, const string& prediction_file);
	HOGFeatureClassifier::TModel OptimizeThresholdsInModel(const string &images_list, const string& model_file);

	void ResponseImage(vector<SlidingRect> &rects, const Mat &image, const string &model_filename);
	void ResponseImage(vector<SlidingRect> &rects, const Mat &image, const HOGFeatureClassifier::TModel &model);
}

//Additional Structures:
struct model;
namespace HOGFeatureClassifier
{
	class TModel {
		auto_ptr<struct model> model_;
		float fastPredictValue;
		float modelThreshold;
		void init()
		{
			fastPredictValue = 0.0f;
			modelThreshold = 0.0f;
		}
	public:
		TModel();
		TModel(struct model* model);
		TModel(const TModel &clone_model);

		TModel& operator=(struct model* model);
		void Save(const string& model_file) const;

		bool Load(const string& model_file);

		struct model* get() const;

		struct model* clone() const;
		void setModelThreshold(float value) { modelThreshold = value; }
		float getModelThreshold() const  { return modelThreshold; }
		void setFastPredictValue(float value) { fastPredictValue = value; }
		float getFastPredictValue() const  { return fastPredictValue; }
	};
}
namespace ImageRecognition
{
	struct SlidingRect
	{
		Rect_<int> rect;
		float value;
		bool falseDetection;

		SlidingRect() { falseDetection = false; }
		SlidingRect(Rect_<int> _rect, float _value)
		{
			rect = _rect;
			value = _value;
			falseDetection = false;
		}
	};
}



#endif //IMAGE_RECOGNITION_H