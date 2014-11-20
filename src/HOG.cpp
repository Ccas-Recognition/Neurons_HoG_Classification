
#include "ImageRecognition.h"
#include "HOG.h"
#include "SlidingWindowOptimizataion.h"
#include <sstream>
#include <time.h>
#include <opencv2/opencv.hpp>

using ImageRecognition::GetRectsFromImage;
using ImageRecognition::FindOptimalThresholdForModel;

namespace HOGFeatureClassifier
{

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
	ifstream stream(data_file.c_str());

	string filename;
	int label;

	int char_idx = data_file.size() - 1;
	for (; char_idx >= 0; --char_idx)
	if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
		break;
	string data_path = data_file.substr(0, char_idx + 1);

	while (!stream.eof() && !stream.fail()) {
		stream >> filename >> label;
		if (filename.size())
			file_list->push_back(make_pair(data_path + filename, label));
	}

	stream.close();
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
	const TLabels& labels,
	const string& prediction_file) {
	// Check that list of files and list of labels has equal size 
	assert(file_list.size() == labels.size());
	// Open 'prediction_file' for writing
	ofstream stream(prediction_file.c_str());

	// Write file names and labels to stream
	for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
		stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
	stream.close();
}


Mat countSobel(const Mat &in)
{
	Mat sobel(in.rows, in.cols, CV_16SC2, Scalar(0,0,0));

	int rows = int(in.rows);
	int cols = int(in.cols);
	#if 1
	for (int i = 0; i < rows; i++)
	for (int j = 0; j < cols; j++) {
		sobel.at<Vec2s>(i, j).val[1] = ((i > 0) ? in.at<uchar>(i - 1, j) : 0) - ((i < rows - 1) ? in.at<uchar>(i + 1, j) : 0);
		sobel.at<Vec2s>(i, j).val[0] = ((j < cols - 1) ? in.at<uchar>(i, j + 1) : 0) - ((j > 0) ? in.at<uchar>(i, j - 1) : 0);
	}
	#else
	for (int i = 1; i < rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			sobel.at<Vec2s>(i, j).val[1] = in.at<uchar>(i - 1, j) - in.at<uchar>(i + 1, j);
			sobel.at<Vec2s>(i, j).val[0] = in.at<uchar>(i, j + 1) - in.at<uchar>(i, j - 1);
		}
	}
	for (int i = 1; i < rows - 1; i++)
	{
		int j = 0;
		sobel.at<Vec2s>(i, j).val[1] = in.at<uchar>(i - 1, j) - in.at<uchar>(i + 1, j);
		sobel.at<Vec2s>(i, j).val[0] = in.at<uchar>(i, j + 1);
		j = cols-1;
		sobel.at<Vec2s>(i, j).val[1] = in.at<uchar>(i - 1, j) - in.at<uchar>(i + 1, j);
		sobel.at<Vec2s>(i, j).val[0] = -in.at<uchar>(i, j - 1);
	}
	for (int j = 1; j < cols - 1; j++)
	{
		int i = 0;
		sobel.at<Vec2s>(i, j).val[1] = - in.at<uchar>(i + 1, j);
		sobel.at<Vec2s>(i, j).val[0] = in.at<uchar>(i, j + 1) - in.at<uchar>(i, j - 1);
		i = rows-1;
		sobel.at<Vec2s>(i, j).val[1] = in.at<uchar>(i - 1, j);
		sobel.at<Vec2s>(i, j).val[0] = in.at<uchar>(i, j + 1) - in.at<uchar>(i, j - 1);
	}
	#endif
	return sobel;
}

// Counting module and direction of gradients (3.3)
Mat countModAndDirOfGrad(const Mat &in, const HOGContext& context, RecognitionStatistics &stat)
{
	Mat sobel = countSobel(in);
	Mat module_direction(sobel.rows, sobel.cols, CV_32FC2);

	for (size_t i = 0; i < sobel.rows; i++)
	for (size_t j = 0; j < sobel.cols; j++) {
		auto x = sobel.at<Vec2s>(i, j).val[0];
		auto y = sobel.at<Vec2s>(i, j).val[1];
		module_direction.at<Vec2f>(i, j).val[0] = sqrtf(float( x*x + y*y ));
		module_direction.at<Vec2f>(i, j).val[1] = atan2(double(y), double(x));
	}

	return module_direction;
}

pair<float, float> phi(float x, float l)
{
	float a(0), b(0);
	if (x > 0) {
		a = cos(l * log(x)) * sqrt(x / cosh(M_PI * l));
		b = -sin(l * log(x)) * sqrt(x / cosh(M_PI * l));
	}
	return make_pair(a, b);
}

float FastPredict(const Mat &modDir)
{
	double sumValue = 0.0;

	for (int y = 1; y < modDir.rows - 1; ++y)
	{
		for (int x = 1; x < modDir.cols - 1; ++x)
		{
			sumValue += modDir.at<Vec2f>(y, x).val[0];
		}
	}
	sumValue /= (modDir.rows - 2)*(modDir.cols - 2);
	return float(sumValue);
}

//static float MAX_DEBUG_VALUE = 0.0f; //DEBUG
void HOG(const int blockSizeX, const int blockSizeY, const int dirSegSize, const Mat &modDir, vector<float> &feats, const HOGContext &context)
{
	vector<float> buffer;
	buffer.reserve(blockSizeX * blockSizeY * dirSegSize + blockSizeX * blockSizeY + 1);

	const int rows = int(modDir.rows); // we use these ...
	const int cols = int(modDir.cols); // ... not only one time
	#if 1
		buffer.resize(blockSizeX * blockSizeY * dirSegSize);
		vector<int> buffer_counts(blockSizeX * blockSizeY);

		// counting hog (3.4)
		for (int i = 1; i < rows-1; i++)
		for (int j = 1; j < cols-1; j++) {
			int blockIndx = int(float( i * blockSizeY ) / rows) * blockSizeX +
				int( float( j * blockSizeX ) / cols);
			int angleIndx = int(((int(modDir.at<Vec2f>(i, j).val[1]) + M_PI) /
				(2 * M_PI)) * dirSegSize);

			int featIndx = blockIndx * dirSegSize + angleIndx;
			buffer[featIndx] += modDir.at<Vec2f>(i, j).val[0];
			buffer_counts[blockIndx] += 1;
		}

		// normalization of histograms (3.5)
		#if 1
			for (int i = 0; i < blockSizeX * blockSizeY; ++i) 
			{
				for (int j = 0; j < dirSegSize; j++)
					buffer[i * dirSegSize + j] /= buffer_counts[i];
			}
		#endif
		if (context.blocksNormalizationType == 1)
		{
			int numOfBlocks = 1;
			for (int i = 0; i < blockSizeX * blockSizeY; i += numOfBlocks) {
				float norm(0);
				for (int j = 0; j < dirSegSize * numOfBlocks; j++)
					norm += pow(buffer[i * dirSegSize + j], 2);

				norm = sqrt(norm);
				for (int j = 0; j < dirSegSize * numOfBlocks; j++)
				if (norm > 0)
					buffer[i * dirSegSize + j] /= norm;
			}
		}
		else
		{
			float norm = 0;
			for (int j = 0; j < buffer.size(); j++)
				norm += buffer[j] * buffer[j];

			norm = sqrt(norm);
			if (norm > 0)
			{
				for (int j = 0; j < buffer.size(); j++)
					buffer[j] /= norm;
			}
		}
	#endif

	#if 0
		vector<double> gradSum(blockSizeX * blockSizeY, 0.0);
		vector<int> gradSumCount(blockSizeX * blockSizeY, 0);
		double sumTotal = 0.0;
		for (int i = 1; i < rows-1; i++)
		{
			for (int j = 1; j < cols-1; j++) 
			{
				int blockIndx = int(float(i * blockSizeY) / rows) * blockSizeX +
					int(float(j * blockSizeX) / cols);
				gradSumCount[blockIndx] += 1;
				gradSum[blockIndx] += double( modDir.at<Vec2f>(i, j).val[0] );
				sumTotal += double(modDir.at<Vec2f>(i, j).val[0]);
			}
		}
		double gradSumMult = 1.0;
		#if 0
			for (int i = 0; i < gradSum.size(); ++i)
				buffer.push_back(float(gradSumMult*gradSum[i] / gradSumCount[i]));
		#endif
			//MAX_DEBUG_VALUE = std::max(MAX_DEBUG_VALUE, float( sumTotal / (rows*cols) ) );//DEBUG
		buffer.push_back(float(gradSumMult*sumTotal / ((rows-2)*(cols-2))));
	#endif
	#if 1
	for (size_t i = 0; i < buffer.size(); i++) {
		for (int j = -context.nonlinear_n; j <= context.nonlinear_n; j++) {
			auto x = phi(buffer[i], j * context.nonlinear_L);
			feats.push_back(x.first);
			feats.push_back(x.second);
		}
	}
	#else
		for (size_t i = 0; i < buffer.size(); i++)
		{
			feats.push_back(buffer[i]);
		}
	#endif
}

void ExtractFeaturesForSample(const Mat& modDir, vector<float> &feats, const HOGContext& context, RecognitionStatistics &stat)
{
	feats.clear();

	const int treeDepth = context.blockSizesX.size();
	for (int i = 0; i < treeDepth; i++)
	{
		HOG(context.blockSizesX[i], context.blockSizesX[i], context.dirSegSize, modDir, feats, context);
	}
}

void FastPredictForSamples(const TFileList& file_list, vector< float > &fastFeatures, const HOGContext& context, RecognitionStatistics &stat)
{
	clock_t begin_time = clock();
	if (stat.flOutputInfo)
		*stat.pInfoStream << "Extract Fast Features";

	fastFeatures.reserve( file_list.size() );
	Mat resizedImage = Mat(context.resizeImageSize, context.resizeImageSize, CV_8UC1);

	for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
	{
		if ((image_idx + 1) % 500 == 0)
		if (stat.flOutputInfo)
			*stat.pInfoStream << ".";
		Mat image;
		image = imread(file_list[image_idx].first.c_str(), 0);
		resize(image, resizedImage, resizedImage.size());
		Mat modDir = countModAndDirOfGrad(resizedImage, context, stat);

		fastFeatures.push_back( FastPredict(modDir) );
		#if 0
		{
			cout << endl << "value: " << fastFeatures.back();
			imshow("modDir", resizedImage);
			char c = waitKey(0); 
		}
		#endif
	}
	//std::cout << "MAX_DEBUG_VALUE: " << MAX_DEBUG_VALUE << endl;//DEBUG
	clock_t end_time = clock();

	if (stat.flOutputInfo)
		*stat.pInfoStream << "done. ";
	if (stat.flOutputTime)
		*stat.pInfoStream << "Fast Feature Extraction Time: " << (float(end_time - begin_time) / 1000.0f) << endl;
}

float FindOptimalFastPredictValues(const TFileList& file_list, const vector< float > &fastFeatures, const HOGContext& context, RecognitionStatistics &stat)
{
	float minValue = 1000.0f;
	float maxValue = 0.0f;

	for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
	{
		#if 0
			cout << fastFeatures[image_idx] << endl;
			imshow("Image", Mat::zeros(100, 100, CV_8U));
			waitKey(0);
		#endif
		if (file_list[image_idx].second == 1)
			minValue = min(minValue, fastFeatures[image_idx]);
		else
			maxValue = max(maxValue, fastFeatures[image_idx]);
	}

	const int MAX_K = 100;
	stat.fastPredictROC.resize(MAX_K);
	stat.fastPredictMinValue = minValue;
	stat.fastPredictMaxValue = maxValue;
	float optimalValue = minValue;
	float optimalError1 = 0.0f;
	float optimalError2 = 0.0f;

	for (int k = 0; k < MAX_K; ++k)
	{
		float current_thr = minValue + float(maxValue - minValue)*float(k) / MAX_K;
		stat.fastPredictROC[k].value = current_thr;
		int error1 = 0;
		int error2 = 0;

		for (size_t i = 0; i < file_list.size(); ++i)
		{
			if ((file_list[i].second == 0) && (fastFeatures[i] > current_thr))
				++error2;
			else if ((file_list[i].second == 1) && (fastFeatures[i] < current_thr))
				++error1;
		}
		float _error1 = float(error1) / file_list.size();
		float _error2 = float(error2) / file_list.size();
		stat.fastPredictROC[k].precision1 = 1.0f - _error1;
		stat.fastPredictROC[k].precision2 = 1.0f - _error2;
		if (_error1 < 0.05f)
		{
			optimalValue = current_thr;
			optimalError1 = _error1;
			optimalError2 = _error2;
		}
	}
	
	if (stat.flOutputInfo)
	{
		*stat.pInfoStream << "Optimal Fast Predict Value: " << optimalValue << " [" << minValue << ", " << maxValue << "]" << endl;
		*stat.pInfoStream << "Fast Predict Error: (" << optimalError1 << ", " << optimalError2 << ")" << endl;
	}
	return optimalValue;
}

// Exatract features from dataset.
void ExtractFeatures(const TFileList& file_list, TFeatures* features, const HOGContext& context, RecognitionStatistics &stat)
{
	clock_t begin_time = clock();
	if (stat.flOutputInfo)
		*stat.pInfoStream << "Extract Features";

	Mat resizedImage = Mat(context.resizeImageSize, context.resizeImageSize, CV_8UC1);
	
	for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx) 
	{
		if ((image_idx + 1) % 500 == 0)
		if (stat.flOutputInfo)
			*stat.pInfoStream << ".";

		Mat image;
		image = imread(file_list[image_idx].first.c_str(), 0);
		#if 1
			resize(image, resizedImage, resizedImage.size());
			Mat modDir = countModAndDirOfGrad(resizedImage, context, stat);
		#else 
			Mat modDir = countModAndDirOfGrad(image, context);
		#endif
		features->push_back(make_pair(vector<float>(), file_list[image_idx].second));
		//features->back().first.reserve(10000);
		ExtractFeaturesForSample(modDir, features->back().first, context, stat);
	}
	if (stat.flOutputInfo)
		*stat.pInfoStream << "done. ";
	//std::cout << "MAX_DEBUG_VALUE: " << MAX_DEBUG_VALUE << endl;//DEBUG
	clock_t end_time = clock();
	if (stat.flOutputTime)
		*stat.pInfoStream << "Extraction Time: " << (float(end_time - begin_time) / 1000.0f) << endl;
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
TModel TrainClassifier(const string& data_file, const string &images_list, const string& model_file, const HOGContext& context, RecognitionStatistics &stat) {
	// List of image file names and its labels
	TFileList file_list;
	// Structure of features of images and its labels
	TFeatures features;
	// Model which would be trained
	TModel model;
	// Parameters of classifier
	TClassifierParams params;
	
	// Load list of image file names and its labels
	LoadFileList(data_file, &file_list);
	// Extract features from images

	vector<float> fastFeatures;
	FastPredictForSamples(file_list, fastFeatures, context, stat);
	model.setFastPredictValue(FindOptimalFastPredictValues(file_list, fastFeatures, context, stat));
	ExtractFeatures(file_list, &features, context, stat);

	// PLACE YOUR CODE HERE
	// You can change parameters of classifier here
	params.C = context.param_C;
	params.solver_type = context.solver_type;
	
	TClassifier classifier(params, stat);
	// Train classifier
	classifier.Train(features, &model);
	// Save model to file
	model.SetContext(context);
	model.Save(model_file);

	if (images_list != "")
	{
		float model_threshold = FindOptimalThresholdForModel( images_list, model, stat );
		model.setModelThreshold(model_threshold);
		model.Save(model_file);
	}
	return model;
}

void PredictData(const string& data_file,
	const string& model_file,
	const string& prediction_file,
	RecognitionStatistics &stat) {

	TFileList file_list;
	TFeatures features;
	TLabels labels;

	TModel model;
	model.Load(model_file);

	LoadFileList(data_file, &file_list);

	ExtractFeatures(file_list, &features, model.GetContext(), stat);

	vector<float> fastFeatures;
	FastPredictForSamples(file_list, fastFeatures, model.GetContext(), stat);

	TClassifier classifier = TClassifier(TClassifierParams(), stat);
	classifier.Predict(features, fastFeatures, model, &labels, file_list, stat);

	SavePredictions(file_list, labels, prediction_file);

	int max_i = 0;
	for (int i = stat.predictROC.size() - 1; i >= 0; --i)
	{
		if ((1.0f - stat.predictROC[i].precision2) < 0.000001f)
			max_i = i;
	}
	model.setModelThreshold( stat.predictROC[max_i].value );
	model.Save(model_file);
}

}