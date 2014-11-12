
#include "HOG.h"
#include <time.h>

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
Mat countModAndDirOfGrad(const Mat &in)
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
//static float MAX_DEBUG_VALUE = 0.0f; //DEBUG
void HOG(const int blockSizeX, const int blockSizeY, const int dirSegSize, const Mat &modDir, vector<float> &feats )
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
		#if 0
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
		#else
				float norm = 0;
				for (int j = 0; j < buffer.size(); j++)
					norm += buffer[j] * buffer[j];

				norm = sqrt(norm);
				if (norm > 0)
				{
					for (int j = 0; j < buffer.size(); j++)
						buffer[j] /= norm;
				}

		#endif
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
		#if 1
			for (int i = 0; i < gradSum.size(); ++i)
				buffer.push_back(float(gradSumMult*gradSum[i] / gradSumCount[i]));
		#endif
			//MAX_DEBUG_VALUE = std::max(MAX_DEBUG_VALUE, float( sumTotal / (rows*cols) ) );//DEBUG
		buffer.push_back(float(gradSumMult*sumTotal / (rows*cols)));
	#endif

	for (size_t i = 0; i < buffer.size(); i++) {
		for (int j = -nonlinear_n; j <= nonlinear_n; j++) {
			auto x = phi(buffer[i], j * nonlinear_L);
			feats.push_back(x.first);
			feats.push_back(x.second);
		}
	}
}

void ExtractFeaturesForSample(const Mat& modDir, vector<float> &feats )
{
	feats.clear();

	const int treeDepth = blockSizeCount;

	for (int i = 0; i < treeDepth; i++)
	{
		HOG(blockSizeX[i], blockSizeY[i], dirSegSize, modDir, feats);
	}
}

// Exatract features from dataset.
void ExtractFeatures(const TFileList& file_list, TFeatures* features)
{
	clock_t begin_time = clock();
	std::cout << "Extract Features";

	Mat resizedImage = Mat(RESIZE_IMAGE_SIZE, RESIZE_IMAGE_SIZE, CV_8UC1);
	
	for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx) 
	{
		if ((image_idx + 1) % 500 == 0)
			cout << ".";

		Mat image;
		image = imread(file_list[image_idx].first.c_str(), 0);
		#if 1
			resize(image, resizedImage, resizedImage.size());
			Mat modDir = countModAndDirOfGrad(resizedImage);
		#else 
			Mat modDir = countModAndDirOfGrad(image);
		#endif
		features->push_back(make_pair(vector<float>(), file_list[image_idx].second));
		//features->back().first.reserve(10000);
		ExtractFeaturesForSample(modDir, features->back().first);
	}
	std::cout << "done" << std::endl;
	//std::cout << "MAX_DEBUG_VALUE: " << MAX_DEBUG_VALUE << endl;//DEBUG
	clock_t end_time = clock();
	cout << "Extraction Time: " << (float(end_time - begin_time)/1000.0f) << endl;
}
// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
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

	ExtractFeatures(file_list, &features);

	// PLACE YOUR CODE HERE
	// You can change parameters of classifier here
	params.C = param_C;
	
	TClassifier classifier(params);
	// Train classifier
	classifier.Train(features, &model);
	// Save model to file
	model.Save(model_file);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
	const string& model_file,
	const string& prediction_file) {
	// List of image file names and its labels
	TFileList file_list;
	// Structure of images and its labels
	//TDataSet data_set;
	// Structure of features of images and its labels
	TFeatures features;
	// List of image labels
	TLabels labels;

	// Load list of image file names and its labels
	LoadFileList(data_file, &file_list);
	// Load images
	//LoadImages(file_list, &data_set);
	// Extract features from images

	ExtractFeatures(file_list, &features);

	// Classifier 
	TClassifier classifier = TClassifier(TClassifierParams());
	// Trained model
	TModel model;
	// Load model from file
	model.Load(model_file);
	// Predict images by its features using 'model' and store predictions
	// to 'labels'
	classifier.Predict(features, model, &labels, file_list);

	// Save predictions
	SavePredictions(file_list, labels, prediction_file);
	// Clear dataset structure
	//ClearDataset(&data_set);
}

}