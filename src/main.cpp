#include "ImageRecognition.h"
#include "argvparser.h"

#include <opencv2/opencv.hpp>
#include <iomanip>
#include <fstream>

using CommandLineProcessing::ArgvParser;
//using ImageRecognition::TModel;
using namespace cv;
using namespace std;

int main(int argc, char** argv) {

	using namespace HOGFeatureClassifier;
	RecognitionStatistics stat;
	stat.flOutputInfo = true;
	stat.flOutputTime = true;
	stat.SetDumpDebugImages(true);

    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue);
	cmd.defineOption("data_set", "File with dataset",
		ArgvParser::OptionRequiresValue);
	cmd.defineOption("context", "Context to HOG training",
		ArgvParser::OptionRequiresValue);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
	cmd.defineOption("images_set", "Images with xml that contain setting information",
		ArgvParser::OptionRequiresValue);
	cmd.defineOption("image", "Image for sliding window",
		ArgvParser::OptionRequiresValue);
	cmd.defineOption("binarization_image", "Binarization Image",
		ArgvParser::OptionRequiresValue);

    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
	cmd.defineOption("sliding_window", "Sliding window");
	cmd.defineOption("binarization", "Binarization");
	cmd.defineOption("optimize_threshold", "Optimize Model Threshold by image_list");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");
	cmd.defineOptionAlternative("images_set", "a");
	cmd.defineOptionAlternative("image", "i");
	cmd.defineOptionAlternative("sliding_window", "s");
	cmd.defineOptionAlternative("optimize_threshold", "o");
	cmd.defineOptionAlternative("binarization_image", "r");
	cmd.defineOptionAlternative("binarization", "b");
	cmd.defineOptionAlternative("context", "c");
        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }
	
	string image_filepath;
	
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");
	bool sliding_window = cmd.foundOption("sliding_window");
	bool optimize_threshold = cmd.foundOption("optimize_threshold");
	bool binarization = cmd.foundOption("binarization");
	
	if (train)
	{
		string model_file = cmd.optionValue("model");
		string data_file = cmd.optionValue("data_set");
		string images_list = "";
		if (cmd.foundOption("images_set"))
			string images_list = cmd.optionValue("images_set");
		
		HOGContext context;
		if (cmd.foundOption("context"))
		{
			string context_file = cmd.optionValue("context");
			ifstream input_context(context_file);
			if (input_context.is_open())
				context.Load(input_context);
		}
		ImageRecognition::TrainHOGClassifier(data_file, images_list, model_file, context, stat);
		#ifdef WIN32
			system("If Not Exist rocs\ mkdir rocs");
			system("del /Q rocs");
		#else
			system("rm -r rocs");
			system("mkdir rocs");
		#endif
		ofstream output("rocs/fastPredictROC.txt");

		if (output.is_open())
		{
			for (int i = 0; i < stat.fastPredictROC.size(); ++i)
			{
				//output << setw(12) << (stat.fastPredictROC[i].value) << " ";
				output << setw(12) << (stat.fastPredictROC[i].falsePositiveRate) << " "
					<< setw(12) << (stat.fastPredictROC[i].truePositiveRate) << endl;
					//<< setw(12) << ((stat.fastPredictROC[i].precision1)) << " "
					//<< setw(12) << ((stat.fastPredictROC[i].precision2)) << endl;
			}
		}
	}
    if (predict) 
	{
		if (!cmd.foundOption("model")) {
			cerr << "Error! Option --model not found!" << endl;
			return 1;
		}
		string model_file = cmd.optionValue("model");
		string data_file = cmd.optionValue("data_set");
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
        string prediction_file = cmd.optionValue("predicted_labels");
		ImageRecognition::PredictData(data_file, model_file, prediction_file, stat);

		#ifdef WIN32
			system("If Not Exist rocs\ mkdir rocs");
			system("del /Q rocs");
		#else
			system("rm -r rocs");
			system("mkdir rocs");
		#endif
		ofstream output("rocs/predictROC.txt");
		if (output.is_open())
		{

			int max_i = 0;
			for (int i = stat.predictROC.size() - 1; i >= 0; --i)
			{
				if ((stat.predictROC[i].falsePositiveRate) < 0.0007f)
					max_i = i;
				//output << setw(12) << ((stat.predictROC[i].value)) << " ";
				output << setw(12) << ((stat.predictROC[i].falsePositiveRate)) << " "
					<< setw(12) << ((stat.predictROC[i].truePositiveRate)) << endl;
			}
			cout << setw(12) << ((stat.predictROC[max_i].falsePositiveRate)) << " "
				<< setw(12) << ((stat.predictROC[max_i].truePositiveRate)) << endl;
		}
    }
	
	if (optimize_threshold)
	{
		if (!cmd.foundOption("model")) {
			cerr << "Error! Option --model not found!" << endl;
			return 1;
		}
		string model_file = cmd.optionValue("model");
		string images_list = cmd.optionValue("images_set");

		ImageRecognition::OptimizeThresholdsInModel(images_list, model_file, stat);
		
		ofstream outputMissings("rocs/recognitionMissing.txt");
		ofstream outputFalseDetections("rocs/recognitionFalseDetections.txt");
		if (outputMissings.is_open() && outputFalseDetections.is_open())
		{
			for (int i = 0; i < stat.recognitionFalseDetections.size(); ++i)
			{
				//output << setw(12) << ((stat.predictROC[i].value)) << " ";
				outputFalseDetections << setw(12) << stat.recognitionFalseDetections[i].value << " "
					<< setw(12) << stat.recognitionFalseDetections[i].error << endl;

				//output << setw(12) << ((stat.predictROC[i].value)) << " ";
				outputMissings << setw(12) << stat.recognitionMissigs[i].value << " "
					<< setw(12) << stat.recognitionMissigs[i].error << endl;
			}
			cout << "missings: " << setw(12) << stat.recognitionMinMissings << ", false_detections: "
				<< setw(12) << stat.recognitionMinFalseDetections << endl;
		}
	}
	
	if (sliding_window)
	{
		if (!cmd.foundOption("model")) {
			cerr << "Error! Option --model not found!" << endl;
			return 1;
		}
		string model_file = cmd.optionValue("model");
		string image_filepath = cmd.optionValue("image");
		Mat image = imread(image_filepath, 0);
		vector< ImageRecognition::SlidingRect > rects;
		ImageRecognition::ResponseImage(rects, image, model_file, stat);
		
		Mat output_image = imread(image_filepath);
		for (int i = 0; i<rects.size(); ++i)
		{
			rectangle(output_image, rects[i].rect, Scalar(0, 255, 0), 1);
		}
		int index = image_filepath.find_last_of('.');
		if (index != -1)
		{
			string output_filepath = image_filepath.substr(0, index);
			output_filepath += "_rects.jpg";
			imwrite(output_filepath, output_image);	
		}
	}
	if (binarization)
	{
		string image_filepath = cmd.optionValue("image");
		string image_output_filepath = cmd.optionValue("binarization_image");
		Mat input_image = imread(image_filepath, 0);
		Mat output_image;
		ImageRecognition::Binarization(input_image, output_image, stat);
		Mat tmp = output_image * 255;
		bitwise_not(tmp, tmp);
		imwrite(image_output_filepath, tmp);
	}
}