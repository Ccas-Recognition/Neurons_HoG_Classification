
#include "ImageRecognition.h"
#include "HOG.h"
#include "SlidingWindowOptimizataion.h"
#include "LaplacianFeatures.h"
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
	const int step = 1;
	#if 1
	for (int i = 0; i < rows; i++)
	for (int j = 0; j < cols; j++) {
		sobel.at<Vec2s>(i, j).val[1] = -((i >= step) ? in.at<uchar>(i - step, j) : 0) + ((i <= rows - step - 1) ? in.at<uchar>(i + step, j) : 0);
		sobel.at<Vec2s>(i, j).val[0] = ((j <= cols - step - 1) ? in.at<uchar>(i, j + step) : 0) - ((j >= step) ? in.at<uchar>(i, j - step) : 0);
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

void buildFeatures(const vector<float> &buffer, vector<float> &feats, const HOGContext &context)
{
	if (context.nonlinear_n > 0)
	{
		for (size_t i = 0; i < buffer.size(); i++)
		{
			for (int j = -context.nonlinear_n; j <= context.nonlinear_n; j++)
			{
				auto x = phi(buffer[i], j * context.nonlinear_L);
				feats.push_back(x.first);
				feats.push_back(x.second);
			}
		}
	}
	else
	{
		for (size_t i = 0; i < buffer.size(); i++)
		{
			feats.push_back(buffer[i]);
		}
	}
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
			int angleIndx = int( (modDir.at<Vec2f>(i, j).val[1] + M_PI) / (2 * M_PI+0.01f) * dirSegSize );
			//cout << angleIndx << " " << modDir.at<Vec2f>(i, j).val[1] << endl;

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
			/*//DEBUG
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
		}*/
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
	buildFeatures(buffer, feats, context);
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

void ExtractLaplacianFeaturesForSample( const Mat& image, vector<float> &feats, const HOGContext& context, RecognitionStatistics &stat, 
	int &result_ceter_x, int &result_ceter_y, float &val )
{
	vector<float> buffer;
	int halfSize = context.lf_halfSizeScanning;
	int center_x = image.cols / 2;
	int center_y = image.rows / 2;

	vector<float> values_col(image.cols);
	vector<float> values_row(image.rows);

	float max_value= 0.0f;
	float max_value_col = 0.0f;
	float max_value_row = 0.0f;

	double mid_value_col = 0.0f;
	double mid_value_row = 0.0f;
	int count = 0;
	for (int dx = -halfSize; dx <= halfSize; ++dx)
	{
		int feat_center_x = center_x + dx;
		for (int y = 0; y < image.rows; ++y)
			values_col[y] = float( image.at<uchar>(y, feat_center_x) );
		
		for (int dy = -halfSize; dy <= halfSize; ++dy)
		{
			int feat_center_y = center_y + dy;
			for (int x = 0; x < image.cols; ++x)
				values_row[x] = float( image.at<uchar>(feat_center_y, x) );

			float value_col = ImageFeatures::ComputeScaleSpaceLaplacianFeature(feat_center_y, values_col, context);
			float value_row = ImageFeatures::ComputeScaleSpaceLaplacianFeature(feat_center_x, values_row, context);

			mid_value_col += value_col;
			mid_value_row += value_row;
			++count;
			if (value_col*value_row > max_value && value_col < 0.0f && value_row < 0.0f)
			{
				max_value = value_col*value_row;
				
				max_value_col = value_col;
				max_value_row = value_row;
				buffer.push_back(value_col);
				buffer.push_back(value_row);
				result_ceter_x = feat_center_x;
				result_ceter_y = feat_center_y;
			}
			else
			{

				buffer.push_back(0.0f);
				buffer.push_back(0.0f);
			}
		}
	}
	double sum = 0.0;
	for (int i = 0; i<buffer.size(); ++i)
		sum += buffer[i];
	for (int i = 0; i < buffer.size(); ++i)
		buffer[i] /= float(sum);

	buildFeatures(buffer, feats, context);
	#if 0
	val = max_value_col*max_value_row;
	#else
	val = float(mid_value_col/count)*float(mid_value_row / count);
	#endif
}

void DebugHOGFeatures(const TFileList& file_list, const HOGContext& _context, RecognitionStatistics &stat)
{
	HOGContext context = _context;
	context.nonlinear_n = 0;
	while (true)
	{
		int index = rand() % file_list.size();
		if (file_list[index].second == 0)
			continue;
		Mat image = imread(file_list[index].first.c_str(), 0);


		Mat _resizedImage = Mat(context.resizeImageSize, context.resizeImageSize, CV_8UC1);
		resize(image, _resizedImage, _resizedImage.size());
		Mat resizedImage;
		GaussianBlur(_resizedImage, resizedImage, Size(3, 3), 0);
		Mat modDir = countModAndDirOfGrad(resizedImage, context, stat);

		vector<float> values;
		int pos_x = 0, pos_y = 0;
		float value;

		string name = file_list[index].first + " " + string((file_list[index].second == 1) ? "fg" : "bg");
		
		ExtractFeaturesForSample(modDir, values, context, stat);
		cout << endl << name << endl << endl;
		float max_value = 0.0f;
		float sum = 0.0f;
		int max_i = 0;
		int index_lim = context.blockSizesX[0] * context.blockSizesY[0] * context.dirSegSize;
		for (int i = 0; i < values.size()+1; ++i)
		{
			if (i % context.dirSegSize == 0 && i>1)
			{
				cout << max_i << " " << sum << endl;
				max_i = 0;
				max_value = 0.0f;
				sum = 0.0f;
				if (i == values.size())
					break;
			}
			sum += values[i];

			if (i == index_lim)
				cout << endl << "-----------------------" << endl;
			
			cout << setw(6) << setprecision(4) << values[i] << " ";
			if (values[i] > max_value)
			{
				max_i = i % context.dirSegSize;
				max_value = values[i];
			}
		}
		//cout << "row: " << values[0] << endl << "col: " << values[1] << endl << "res: " << (values[0] * values[1]) << endl << endl;
		//cout << "value: " << value << endl << endl;

		Mat blur, otsu;
		GaussianBlur(resizedImage, blur, Size(3, 3), 0);
		threshold(blur, otsu, 0, 255, THRESH_BINARY | THRESH_OTSU);

		const int scale = 20;
		Mat outputImage(resizedImage.rows*scale, resizedImage.cols*scale, CV_8UC3);
		Mat outputImage2(resizedImage.rows*scale, resizedImage.cols*scale, CV_8UC3);
		resize(otsu, outputImage2, outputImage2.size(), 0.0, 0.0, CV_INTER_NN);

		Mat tmpImage;
		cvtColor(resizedImage, tmpImage, CV_GRAY2RGB);

		resize(tmpImage, outputImage, outputImage.size(), 0.0, 0.0, CV_INTER_NN);
		for (int i = context.blockSizesX.size()-1; i >= 0; --i)
		{
			int sz = context.blockSizesX[i];
			for (int jx = 0; jx < sz-1; ++jx)
			{
				int pos_x = int( (1.0f / sz)*(jx+1)*outputImage.cols );
				line(outputImage, Point(pos_x, 0), Point(pos_x, outputImage.rows - 1), (i == 0) ? Scalar(0, 255, 0) : Scalar(0, 0, 255));
			}
			for (int jy = 0; jy < sz - 1; ++jy)
			{
				int pos_y = int((1.0f / sz)*(jy + 1)*outputImage.rows);
				line(outputImage, Point(0, pos_y), Point(outputImage.cols-1, pos_y), (i == 0) ? Scalar(0, 255, 0) : Scalar(0, 0, 255));
			}
		}
		imshow(name, outputImage);
		moveWindow(name, 1000, 20);
		imshow(name + "2", outputImage2);
		moveWindow(name + "2", 1000, 540);

		vector<Vec3b> colors(context.dirSegSize);
		for (int i = 0; i < colors.size(); ++i)
		{
			colors[i].val[0] = uchar(float(rand()) / RAND_MAX * 255);
			colors[i].val[1] = uchar(float(rand()) / RAND_MAX * 255);
			colors[i].val[2] = uchar(float(rand()) / RAND_MAX * 255);
		}
		Mat gradImage(resizedImage.rows, resizedImage.cols, CV_8UC3);
		for (int y = 0; y<resizedImage.rows; ++y)
		for (int x = 0; x < resizedImage.cols; ++x)
		{
			float px = cosf(modDir.at<Vec2f>(y, x).val[1]);
			float py = sinf(modDir.at<Vec2f>(y, x).val[1]);
			int angleIndx = int((modDir.at<Vec2f>(y, x).val[1] + M_PI) / (2 * M_PI + 0.01f) * context.dirSegSize);
			//gradImage.at<Vec3b>(y, x) = colors[angleIndx];
			gradImage.at<Vec3b>(y, x) = Vec3b(0, uchar((px + 1.0f)*127.5f), uchar((py + 1.0f)*127.5f));
		}
		Mat outputGradImage(resizedImage.rows*scale, resizedImage.cols*scale, CV_8UC3);
		resize(gradImage, outputGradImage, outputGradImage.size(), 0.0, 0.0, CV_INTER_NN);
		imshow(name + "3", outputGradImage);
		moveWindow(name + "3", 500, 540);
		/*
		MSER mser(1, 1, 50, .25, .2, 200, 1.01, 0.003, 5);
		vector<vector<Point>> points;
		mser(resizedImage, points);

		Mat mserImage(resizedImage.rows, resizedImage.cols, CV_8UC3);
		for (int y = 0; y<resizedImage.rows; ++y)
		for (int x = 0; x<resizedImage.cols; ++x)
			mserImage.at<Vec3b>(y, x) = Vec3b(resizedImage.at<uchar>(y, x), resizedImage.at<uchar>(y, x), resizedImage.at<uchar>(y, x));
		for (int i = 0; i < (int)points.size(); i++)
		{
			uchar r = uchar(float(rand()) / RAND_MAX * 255);
			uchar g = uchar(float(rand()) / RAND_MAX * 255);
			uchar b = uchar(float(rand()) / RAND_MAX * 255);
			for (int j = 0; j < (int)points.at(i).size(); j++)
			{
				Point p = points.at(i).at(j);
				mserImage.at<Vec3b>(p) = Vec3b(b, g, r);
			}
		}
		Mat outputMserImage(resizedImage.rows*scale, resizedImage.cols*scale, CV_8UC3);
		resize(mserImage, outputMserImage, outputMserImage.size(), 0.0, 0.0, CV_INTER_NN);
		imshow(name + "3", outputMserImage);
		moveWindow(name + "3", 500, 540);
		*/
		waitKey(0);
		destroyAllWindows();
	}
}

void DebugLaplacianFeatures(const TFileList& file_list, const HOGContext& context, RecognitionStatistics &stat)
{
	while (true)
	{
		int index = rand() % file_list.size();
		Mat image = imread(file_list[index].first.c_str(), 0);
		
		Mat resizedImage = Mat(context.resizeImageSize, context.resizeImageSize, CV_8UC1);
		resize(image, resizedImage, resizedImage.size());
		
		vector<float> values;
		int pos_x = 0, pos_y = 0;
		float value;
		
		string name = file_list[index].first + " " + string((file_list[index].second == 1) ? "fg" : "bg");
		ExtractLaplacianFeaturesForSample(resizedImage, values, context, stat, pos_x, pos_y, value);
		cout << name << endl;
		//cout << "row: " << values[0] << endl << "col: " << values[1] << endl << "res: " << (values[0] * values[1]) << endl << endl;
		cout << "value: " << value << endl << endl;
		
		const int scale = 20;
		Mat outputImage(resizedImage.rows*scale, resizedImage.cols*scale, CV_8UC3);
		Mat tmpImage;
		cvtColor(resizedImage, tmpImage, CV_GRAY2RGB);
		
		resize(tmpImage, outputImage, outputImage.size(), 0.0, 0.0, CV_INTER_NN);

		line(outputImage, Point(0, pos_y*scale), Point(outputImage.cols, pos_y*scale), Scalar(0, 0, 255));
		line(outputImage, Point(pos_x*scale, 0), Point(pos_x*scale, outputImage.rows), Scalar(0, 0, 255));
		
		imshow(name, outputImage);
		moveWindow(name, 1000, 200);
		waitKey(0);
		destroyAllWindows();
	}
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
		resize(image, resizedImage, resizedImage.size());

		features->push_back(make_pair(vector<float>(), file_list[image_idx].second));
		if (context.features_type == 0)
		{
			#if 1
				Mat modDir = countModAndDirOfGrad(resizedImage, context, stat);
			#else 
				Mat modDir = countModAndDirOfGrad(image, context);
			#endif
			ExtractFeaturesForSample(modDir, features->back().first, context, stat);

			int pos_x, pos_y;
			float value;
			ExtractLaplacianFeaturesForSample(resizedImage, features->back().first, context, stat, pos_x, pos_y, value);
		}
		else
		{
			int pos_x, pos_y;
			float value;
			ExtractLaplacianFeaturesForSample(resizedImage, features->back().first, context, stat, pos_x, pos_y, value);
		}
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
	#if 0
		DebugHOGFeatures(file_list, context, stat);
	#endif
	#if 0
		DebugLaplacianFeatures(file_list, context, stat);
	#endif

	if (context.useFastFeatures)
	{
		vector<float> fastFeatures;
		FastPredictForSamples(file_list, fastFeatures, context, stat);
		model.setFastPredictValue(FindOptimalFastPredictValues(file_list, fastFeatures, context, stat));
	}
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
	if (model.GetContext().useFastFeatures)
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