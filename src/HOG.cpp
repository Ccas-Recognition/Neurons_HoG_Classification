
#include "HOG.h"

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

// Making grayscale image (3.1)
Mat toGreyScale(const Mat &in)
{
	Mat img(in.rows, in.cols, CV_8UC1);
	for (int i = 0; i < in.rows; i++)
	for (int j = 0; j < in.cols; j++) {
		img.at<uchar>(i, j) = int(0.299 * in.at<Vec3b>(i, j).val[2] +
			0.587 * in.at<Vec3b>(i, j).val[1] +
			0.114 * in.at<Vec3b>(i, j).val[0] );
	}

	return img;
}

// Making matrixes of x and y parts of gradient (3.2)
// Sobel by unary map gives worse result at all
// so here hand-made sobel filter
Mat countSobel(const Mat &in)
{
	Mat img = toGreyScale(in);
	Mat sobel(in.rows, in.cols, CV_16SC2);

	int rows = int(in.rows);
	int cols = int(in.cols);
	for (int i = 0; i < rows; i++)
	for (int j = 0; j < cols; j++) {
		sobel.at<Vec2s>(i, j).val[1] = ((i > 0) ? img.at<uchar>(i - 1, j) : 0) - ((i < rows - 1) ? img.at<uchar>(i + 1, j) : 0);
		sobel.at<Vec2s>(i, j).val[0] = ((j < cols - 1) ? img.at<uchar>(i, j + 1) : 0) - ((j > 0) ? img.at<uchar>(i, j - 1) : 0);
	}

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
		module_direction.at<Vec2f>(i, j).val[0] = sqrt(x*x + y*y);
		module_direction.at<Vec2f>(i, j).val[1] = atan2(y, x);
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

vector<float> HOG(const int blockSizeX, const int blockSizeY, const int dirSegSize, const Mat &image)
{
	vector<float> one_image_features(blockSizeX * blockSizeY * dirSegSize, 0);

	Mat modDir = countModAndDirOfGrad(image);

	// counting hog (3.4)
	const int rows = int(modDir.rows); // we use these ...
	const int cols = int(modDir.cols); // ... not only one time
	for (int i = 0; i < rows; i++)
	for (int j = 0; j < cols; j++) {
		int blockIndx = int(float( i * blockSizeY ) / rows) * blockSizeX +
			int( float( j * blockSizeX ) / cols);
		int angleIndx = int(((int(modDir.at<Vec2f>(i, j).val[1]) + M_PI) /
			(2 * M_PI)) * dirSegSize);

		int featIndx = blockIndx * dirSegSize + angleIndx;
		one_image_features[featIndx] += modDir.at<Vec2f>(i, j).val[0];
	}

	// normalization of histograms (3.5)
	int numOfBlocks(2); // normalizing blocks with norm of numOfBlocks blocks
	for (int i = 0; i < blockSizeX * blockSizeY; i += numOfBlocks) {
		float norm(0);
		for (int j = 0; j < dirSegSize * numOfBlocks; j++)
			norm += pow(one_image_features[i * dirSegSize + j], 2);

		norm = sqrt(norm);
		for (int j = 0; j < dirSegSize * numOfBlocks; j++)
		if (norm > 0) {
			one_image_features[i * dirSegSize + j] /= norm;
		}
	}

#if 0
	const int blockSX(8);
	const int blockSY(8);
	vector<int> colorR(blockSX * blockSY, 0);
	vector<int> colorG(blockSX * blockSY, 0);
	vector<int> colorB(blockSX * blockSY, 0);
	vector<int> colNum(blockSX * blockSY, 0);

	for (int i = 0; i < rows; i++)
	for (int j = 0; j < cols; j++) {
		int blockIndx = int(i * blockSY / rows) * blockSX +
			int(j * blockSX / cols);
		colorR[blockIndx] += (*image)(j, i)->Red;
		colorG[blockIndx] += (*image)(j, i)->Green;
		colorB[blockIndx] += (*image)(j, i)->Blue;
		colNum[blockIndx]++;
	}

	for (size_t i = 0; i < colorR.size(); i++)
	{
		one_image_features.push_back(colorR[i] / (255 * colNum[i]));
		one_image_features.push_back(colorG[i] / (255 * colNum[i]));
		one_image_features.push_back(colorB[i] / (255 * colNum[i]));
	}
#endif

	vector<float> tmp;

	for (size_t i = 0; i < one_image_features.size(); i++) {
		for (int j = -nonlinear_n; j <= nonlinear_n; j++) {
			auto x = phi(one_image_features[i], j * nonlinear_L);
			tmp.push_back(x.first);
			tmp.push_back(x.second);
		}
	}

	one_image_features.clear();
	return tmp;
}

// Exatract features from dataset.
void ExtractFeatures(const TFileList& file_list, TFeatures* features)
{
	const int treeDepth(blockSizeX.size());
	for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx) {
		if ((image_idx + 1) % 500 == 0)
			cout << ".";

		vector<float> one_image_features;
		for (int i = 0; i < treeDepth; i++)
		{
			Mat image;
			// Read image from file
			image = imread( file_list[image_idx].first.c_str() );
			// Add image and it's label to dataset
			//data_set->push_back(make_pair(image, file_list[image_idx].second));

			auto tmp = HOG(blockSizeX[i], blockSizeY[i], dirSegSize, image);
			for (size_t k = 0; k < tmp.size(); k++) {
				one_image_features.push_back(tmp[k]);
			}
		}
		features->push_back(make_pair(one_image_features, file_list[image_idx].second));
	}
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
	std::cout << "Extract Features";
	ExtractFeatures(file_list, &features);
	std::cout << "done" << std::endl;

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
	std::cout << "Extract Features";
	ExtractFeatures(file_list, &features);
	std::cout << " done" << std::endl;

	// Classifier 
	TClassifier classifier = TClassifier(TClassifierParams());
	// Trained model
	TModel model;
	// Load model from file
	model.Load(model_file);
	// Predict images by its features using 'model' and store predictions
	// to 'labels'
	classifier.Predict(features, model, &labels);

	// Save predictions
	SavePredictions(file_list, labels, prediction_file);
	// Clear dataset structure
	//ClearDataset(&data_set);
}

}