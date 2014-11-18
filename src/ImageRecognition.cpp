#include "ImageRecognition.h"
#include "HOG.h"
#include "linear.h"
#include "classifier.h"
#include "HOG_Functor.h"
#include "utils.h"
#include "SlidingWindowOptimizataion.h"

using HOGFeatureClassifier::HOGContext;

namespace ImageRecognition
{
	HOGFeatureClassifier::TModel TrainHOGClassifier(const string& data_file, const string &images_list, const string& model_file, const HOGContext& context, RecognitionStatistics &stat)
	{
		return HOGFeatureClassifier::TrainClassifier(data_file, images_list, model_file, context, stat);
	}

	void PredictData(const string& data_file, const string& model_file, const string& prediction_file,	RecognitionStatistics &stat)
	{
		HOGFeatureClassifier::PredictData(data_file, model_file, prediction_file, stat);
	}

	HOGFeatureClassifier::TModel OptimizeThresholdsInModel(const string &images_list, const string& model_file, RecognitionStatistics &stat)
	{
		TModel model;
		model.Load(model_file);

		float model_threshold = ImageRecognition::FindOptimalThresholdForModel(images_list, model, stat);
		model.setModelThreshold(model_threshold);
		model.Save(model_file);
		return model;
	}
}

namespace HOGFeatureClassifier
{
	void HOGContext::Load(istream &input)
	{
		using namespace Utils;

		blockSizesX.clear();
		blockSizesY.clear();
		slidingWindowSizes.clear();
		blockSizesX.reserve(8);
		blockSizesY.reserve(8);
		slidingWindowSizes.reserve(8);

		while (!input.eof())
		{
			string variable, values, value;
			input >> variable;
			if (variable == "blockSizesX")
			{
				getline(input, values);
				stringstream ss(values);
				while (getline(ss, value, ' '))
				if (value != "")
						blockSizesX.push_back(str2int(value));
			}
			else if (variable == "blockSizesY")
			{
				getline(input, values);
				stringstream ss(values);
				while (getline(ss, value, ' '))
				if (value != "")
					blockSizesY.push_back(str2int(value));
			}
			else if (variable == "slidingWindowSizes")
			{
				getline(input, values);
				stringstream ss(values);
				while (getline(ss, value, ' '))
				if (value != "")
					slidingWindowSizes.push_back(str2int(value));
			}
			else if (variable == "dirSegSize")
				input >> dirSegSize;
			else if (variable == "resizeImageSize")
				input >> resizeImageSize;
			else if (variable == "nonlinear_n")
				input >> nonlinear_n;
			else if (variable == "nonlinear_L")
				input >> nonlinear_L;
			else if (variable == "param_C")
				input >> param_C;
			else if (variable == "standartSlidingWindowSize")
				input >> standartSlidingWindowSize;
			else if (variable == "standartslidingWindowStep")
				input >> standartslidingWindowStep;
		}
	}
	void HOGContext::Save(ostream &output) const
	{
		output << "blockSizesX ";
		for (int i = 0; i < blockSizesX.size(); ++i)
			output << blockSizesX[i] << " ";
		output << endl << "blockSizesY ";
		for (int i = 0; i < blockSizesY.size(); ++i)
			output << blockSizesY[i] << " ";

		output << endl << "slidingWindowSizes ";
		for (int i = 0; i < slidingWindowSizes.size(); ++i)
			output << slidingWindowSizes[i] << " ";

		output << endl << "dirSegSize " << dirSegSize;
		output << endl << "resizeImageSize " << resizeImageSize;
		output << endl << "nonlinear_n " << nonlinear_n;
		output << endl << "nonlinear_L " << nonlinear_L;
		output << endl << "param_C " << param_C;
		output << endl << "standartSlidingWindowSize " << standartSlidingWindowSize;
		output << endl << "standartslidingWindowStep " << standartslidingWindowStep;
	}

	TModel::TModel() : model_(NULL) { init(); }
	
	TModel::TModel(struct model* model) : model_(model) { init(); }
	
	TModel::TModel(const TModel &clone_model) : model_(clone_model.clone())
	{
		init();
		fastPredictValue = clone_model.fastPredictValue;
		modelThreshold = clone_model.modelThreshold;
	}

	TModel& TModel::operator=(struct model* model) 
	{
		init();
		model_ = auto_ptr<struct model>(model);
		return *this;
	}
	void TModel::Save(const string& model_file) const 
	{
		assert(model_.get());
		save_model(model_file.c_str(), model_.get());
		ofstream additionalInfo(model_file, std::ofstream::out | std::ofstream::app);
		additionalInfo << endl;
		additionalInfo << "fast_predict_value " << fastPredictValue << endl;
		additionalInfo << "model_threshold " << modelThreshold << endl;
		additionalInfo << "Context:" << endl;
		context.Save(additionalInfo);
	}	
	// Load model from file
	bool TModel::Load(const string& model_file) 
	{
		using namespace std;
		model_ = auto_ptr<struct model>(load_model(model_file.c_str()));

		if (model_.get() == 0)
			return false;

		ifstream additionalInfo(model_file);
		while (!additionalInfo.eof())
		{
			string line;
			getline(additionalInfo, line);
			stringstream ss(line);
			string command;
			getline(ss, command, ' ');
			if (command == "fast_predict_value")
				ss >> fastPredictValue;
			else if (command == "model_threshold")
				ss >> modelThreshold;
			else if (command == "Context:")
			{
				context.Load(additionalInfo);
				break;
			}
		}

		return true;
	}
	// Get pointer to liblinear model
	struct model* TModel::get() const
	{
		return model_.get();
	}

	struct model* TModel::clone() const 
	{
		return new struct model(*(model_.get()));
	}
}