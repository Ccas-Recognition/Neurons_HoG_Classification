#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include <vector>
#include <string>
#include <cstdlib>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <cassert>

#include "linear.h"
#include "ImageRecognition.h"
#include "consts.h"

using namespace std;

namespace HOGFeatureClassifier
{
	typedef vector<pair<vector<float>, int> > TFeatures;
	typedef vector<int> TLabels;

	// Parameters for classifier training
	// Read more about it in liblinear documentation
	struct TClassifierParams {
		double bias;
		int solver_type;
		double C;
		double eps;
		int nr_weight;
		int* weight_label;
		double* weight;

		TClassifierParams() {
			bias = -1;
			solver_type = L2R_L2LOSS_SVC;//solver_type = L2R_L2LOSS_SVC_DUAL;
			C = 0.1;
			eps = 1e-4;
			nr_weight = 0;
			weight_label = NULL;
			weight = NULL;
		}
	};

	// Classifier. Encapsulates liblinear classifier.
	class TClassifier {
		// Parameters of classifier
		TClassifierParams params_;
		RecognitionStatistics &stat;
	public:
		// Basic constructor
		TClassifier(const TClassifierParams& params, RecognitionStatistics &_stat) : params_(params), stat(_stat) {}

		// Train classifier
		void Train(const TFeatures& features, TModel* model) {
			// Number of samples and features must be nonzero
			size_t number_of_samples = features.size();
			assert(number_of_samples > 0);

			size_t number_of_features = features[0].first.size();
			assert(number_of_features > 0);

			if (stat.flOutputInfo)
			{
				std::cout << "number of samples: " << number_of_samples << std::endl;
				std::cout << "number of features: " << number_of_features << std::endl;
			}

			vector<struct feature_node> data(number_of_samples*(number_of_features + 1));
			//std::cout << "Setting up prob" << std::endl;
			// Description of one problem

			struct problem prob;
			prob.l = number_of_samples;
			prob.bias = -1;
			prob.n = number_of_features;
			prob.y = new double[number_of_samples];
			prob.x = new struct feature_node*[number_of_samples];

			// Fill struct problem
			for (size_t sample_idx = 0; sample_idx < number_of_samples; ++sample_idx)
			{
				//prob.x[sample_idx] = new struct feature_node[number_of_features + 1];
				prob.x[sample_idx] = &(data[sample_idx*(number_of_features + 1)]);
				for (unsigned int feature_idx = 0; feature_idx < number_of_features; feature_idx++)
				{
					prob.x[sample_idx][feature_idx].index = feature_idx + 1;
					prob.x[sample_idx][feature_idx].value = features[sample_idx].first[feature_idx];
				}
				prob.x[sample_idx][number_of_features].index = -1;
				prob.y[sample_idx] = features[sample_idx].second;
			}

			// Fill param structure by values from 'params_'
			struct parameter param;
			param.solver_type = params_.solver_type;
			param.C = params_.C;      // try to vary it
			param.eps = params_.eps;
			param.nr_weight = params_.nr_weight;
			param.weight_label = params_.weight_label;
			param.weight = params_.weight;

			// Train model
			//std::cout << "Train begin" << std::endl;
			*model = train(&prob, &param);
			//std::cout << "Train end" << std::endl;

			// Clear param structure
			destroy_param(&param);
			// clear problem structure
			delete[] prob.y;
			// for (unsigned int sample_idx = 0; sample_idx < number_of_samples; ++sample_idx)
			//    delete[] prob.x[sample_idx];
			delete[] prob.x;
		}

		static void ConvertFeaturesToClassifierType(const vector<float>& features, vector< struct feature_node > &classifier_features, RecognitionStatistics &stat)
		{
			classifier_features.clear();
			for (int i = 0; i < features.size(); ++i)
			{
				classifier_features.push_back(feature_node());
				classifier_features.back().index = i + 1;
				classifier_features.back().value = features[i];
			}
			classifier_features.push_back(feature_node());
			classifier_features.back().index = -1;
		}

		static double ComputePredictValue(const vector< struct feature_node > &classifier_features, const TModel& model, RecognitionStatistics &stat)
		{
			double arr[1];
			predict_values(model.get(), &(classifier_features[0]), arr);
			return arr[0];
		}

		static void ROC_Curve(const vector<float> &values, const vector<pair<string, int> > &file_list, RecognitionStatistics &stat)
		{
			float max_value = 0.0f;
			float min_value = 0.0f;
			double sum_value = 0.0;
			for (int i = 0; i < values.size(); ++i)
			{
				max_value = max(max_value, values[i]);
				min_value = min(min_value, values[i]);
				sum_value += double( values[i] );
			}
			sum_value /= values.size();
			if (stat.flOutputInfo)
			{
				cout << "min: " << min_value << endl;
				cout << "max: " << max_value << endl;
				//cout << "avg: " << sum_value << endl;
			}
			int iters = 100;
			for (int k = 0; k < iters; ++k)
			{
				if ((k > 1) && (k < (iters / 5)) )
					continue;
				float t = max_value*float(k) / (iters);
				int error1 = 0; //1 - right, 0 - predicted
				int error2 = 0; //0 - right, 1 - predicted
				for (int i = 0; i < values.size(); ++i)
				{
					int predict_value = int(values[i] > t);
					if (file_list[i].second != predict_value)
					{
						if (file_list[i].second == 1)
							++error1;
						else
							++error2;
					}
				}
				float precision = 100 * float(values.size() - error1 - error2) / values.size();
				float precision_error1 = 100 * (float(error1) / values.size());
				float precision_error2 = 100 * (float(error2) / values.size());
				printf("%10f (%10f, %10f): %10f\n", precision, precision_error1, precision_error2, t);
				if (precision_error2 < 0.00001f)
					break;
				//cout << setprecision(4) << precision << " (" << precision_error1 << ", " << precision_error2 << ")" << ": " << t << endl;
				//cout << "Precision: " << (precision*100.0f) << "%" << endl;
				//cout << "Error 1 (1 - right, 0 - predicted): " <<  << "%" << endl;
				//cout << "Error 2 (0 - right, 1 - predicted): " << (float(error2) / values.size() * 100) << "%" << endl;
			}
		}
		
		// Predict data
		static void Predict(const TFeatures& features, const TModel& model, TLabels* labels, const vector<pair<string, int> > &file_list, RecognitionStatistics &stat)
		{
			// Number of samples and features must be nonzero
			size_t number_of_samples = features.size();
			assert(number_of_samples > 0);
			size_t number_of_features = features[0].first.size();
			assert(number_of_features > 0);

			// Fill struct problem
			vector< struct feature_node > x;
			vector< float > values;
			values.reserve(features.size());
			x.reserve(number_of_features + 1);

			std::ofstream outfile("dump.txt");
			for (size_t sample_idx = 0; sample_idx < features.size(); ++sample_idx) 
			{
				ConvertFeaturesToClassifierType(features[sample_idx].first, x, stat);
				//labels->push_back( int( predict( model.get(), &(x[0]) ) ) );
				values.push_back(ComputePredictValue(x, model, stat));
				labels->push_back(int(values.back() > 0));
				outfile << labels->back() << " " << ComputePredictValue(x, model, stat) << std::endl;
			}

			ROC_Curve(values, file_list, stat);
		}
	};
}//HOGFeatureClassifier
#endif