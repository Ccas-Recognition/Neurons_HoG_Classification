#include "ImageRecognition.h"
#include "HOG.h"
#include "linear.h"
#include "classifier.h"
#include "HOG_Functor.h"
#include "SlidingWindowOptimizataion.h"

namespace ImageRecognition
{
	HOGFeatureClassifier::TModel TrainHOGClassifier(const string& data_file, const string &images_list, const string& model_file)
	{
		return HOGFeatureClassifier::TrainClassifier(data_file, images_list, model_file);
	}

	void PredictData(const string& data_file, const string& model_file, const string& prediction_file)
	{
		HOGFeatureClassifier::PredictData(data_file, model_file, prediction_file);
	}

	HOGFeatureClassifier::TModel OptimizeThresholdsInModel(const string &images_list, const string& model_file)
	{
		TModel model;
		model.Load(model_file);

		float model_threshold = ImageRecognition::FindOptimalThresholdForModel(images_list, model);
		model.setModelThreshold(model_threshold);
		model.Save(model_file);
		return model;
	}
}

namespace HOGFeatureClassifier
{
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
		additionalInfo << endl << "fast_predict_value " << fastPredictValue << endl;
		additionalInfo << endl << "model_threshold " << modelThreshold << endl;
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