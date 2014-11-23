#ifndef IMAGE_RECOGNITION_H
#define IMAGE_RECOGNITION_H

#include<opencv2/core/core.hpp>

#include <vector>
#include <string>
#include <cassert>
#include <ostream>
#include <istream>
#include <iostream>

using namespace std;
using namespace cv;

//Forward Declaration:
namespace HOGFeatureClassifier { class TModel; }
namespace HOGFeatureClassifier { class HOGContext; }
namespace HOGFeatureClassifier { class RecognitionStatistics; }
namespace ImageRecognition { class SlidingRect; }
using HOGFeatureClassifier::HOGContext;
using HOGFeatureClassifier::RecognitionStatistics;

//Main Interface:
namespace ImageRecognition
{
	HOGFeatureClassifier::TModel TrainHOGClassifier(const string& data_file, const string &images_list, const string& model_file, const HOGContext& context, RecognitionStatistics &stat);
	void PredictData(const string& data_file, const string& model_file, const string& prediction_file, RecognitionStatistics &stat);
	HOGFeatureClassifier::TModel OptimizeThresholdsInModel(const string &images_list, const string& model_file, RecognitionStatistics &stat );

	void ResponseImage(vector<SlidingRect> &rects, const Mat &image, const string &model_filename, RecognitionStatistics &stat);
	void ResponseImage(vector<SlidingRect> &rects, const Mat &image, const HOGFeatureClassifier::TModel &model, RecognitionStatistics &stat);
}

//Additional Structures:
struct model;
namespace HOGFeatureClassifier
{
	struct HOGContext
	{
		//HOG:
		vector<int> blockSizesX, blockSizesY;
		int dirSegSize;
		int resizeImageSize;
		float nonlinear_n;
		float nonlinear_L;
		double param_C;

		//Sliding Window
		vector<int> slidingWindowSizes;
		int standartSlidingWindowSize;
		int standartslidingWindowStep;
		int blocksNormalizationType;
		int solver_type;
		
		int lf_halfSizeScanning;
		float lf_begin_scale;
		float lf_end_scale;
		float lf_scale_step;

		bool useFastFeatures;
		int features_type; // 0 - HOG, 1 - cross features

		HOGContext()
		{
			int _blockSizes[2] = { 3, 6 };
			blockSizesX.resize(sizeof(_blockSizes) / sizeof(int));
			blockSizesY.resize(sizeof(_blockSizes) / sizeof(int));
			copy(_blockSizes, _blockSizes + sizeof(_blockSizes) / sizeof(int), blockSizesX.begin());
			copy(_blockSizes, _blockSizes + sizeof(_blockSizes) / sizeof(int), blockSizesY.begin());

			dirSegSize = 8;
			resizeImageSize = 24;
			nonlinear_n = 2;
			nonlinear_L = 0.5f;
			param_C = 0.08;

			//Sliding Window
			int _slidingWindowSizes[] = { 24, 36, 48, 60 };
			slidingWindowSizes.resize(sizeof(_slidingWindowSizes) / sizeof(int));
			copy( _slidingWindowSizes, _slidingWindowSizes + 4, slidingWindowSizes.begin() );
			standartSlidingWindowSize = 24;
			standartslidingWindowStep = 4;
			blocksNormalizationType = 1;
			solver_type = 2;

			features_type = 0;
			useFastFeatures = true;
			lf_halfSizeScanning = 4;

			float lf_begin_scale = 0.1f;
			float lf_end_scale = 0.3f;
			float lf_scale_step = 0.05;
		}
		void Load(istream &input);
		void Save(ostream &output)const;
	};

	struct ROCValue
	{
		float truePositiveRate;
		float falsePositiveRate;
		float value;
	};

	struct RecognitionStatistics
	{
		//options
		bool flOutputInfo;
		bool flOutputTime;
		bool flDumpDebugImages;

		vector<ROCValue> fastPredictROC;
		float fastPredictMinValue;
		float fastPredictMaxValue;

		vector<ROCValue> predictROC;
		float predictMinValue;
		float predictMaxValue;

		ostream* pInfoStream;

		RecognitionStatistics()
		{
			//options
			flOutputInfo = false;
			flOutputTime = false;
			flDumpDebugImages = false;

			pInfoStream = &cout;

			fastPredictMinValue = 0.0f;
			fastPredictMaxValue = 0.0f;
			predictMinValue = 0.0f;
			predictMaxValue = 0.0f;
		}

		void SetDumpDebugImages(bool on)
		{
			flDumpDebugImages = on;
			if (on)
			{
				#ifdef WIN32
					system("If Not Exist dump\ mkdir dump");
					system("del /Q dump");
				#else
					system("rm -r dump");
					system("mkdir dump");
				#endif
			}
		}
	};

	class TModel {
		auto_ptr<struct model> model_;
		float fastPredictValue;
		float modelThreshold;
		HOGContext context;
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
		void SetContext(const HOGContext& _context) { context = _context; }
		const HOGContext& GetContext() const { return context; }
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