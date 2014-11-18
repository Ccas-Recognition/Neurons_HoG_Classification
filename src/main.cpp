#include "ImageRecognition.h"
#include "argvparser.h"

#include <opencv2/opencv.hpp>
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
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
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

    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
	cmd.defineOption("sliding_window", "Sliding window");
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
	cmd.defineOptionAlternative("context", "c");
        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }
	
	string image_filepath;
	string model_file = cmd.optionValue("model");

    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");
	bool sliding_window = cmd.foundOption("sliding_window");
	bool optimize_threshold = cmd.foundOption("optimize_threshold");

	if (train)
	{
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
	}
    if (predict) 
	{
		string data_file = cmd.optionValue("data_set");
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
        string prediction_file = cmd.optionValue("predicted_labels");
		ImageRecognition::PredictData(data_file, model_file, prediction_file, stat);
    }
	
	if (optimize_threshold)
	{
		string images_list = cmd.optionValue("images_set");

		ImageRecognition::OptimizeThresholdsInModel(images_list, model_file, stat);
	}
	
	if (sliding_window)
	{
		string image_filepath = cmd.optionValue("image");
		Mat image = imread(image_filepath, 0);
		vector< ImageRecognition::SlidingRect > rects;
		ImageRecognition::ResponseImage(rects, image, model_file, stat);
		
		Mat output_image = imread(image_filepath);
		for (int i = 0; i<rects.size(); ++i)
		{
			rectangle(output_image, rects[i].rect, Scalar(0, 0, 255), 1);
		}
		int index = image_filepath.find_last_of('.');
		if (index != -1)
		{
			string output_filepath = image_filepath.substr(0, index);
			output_filepath += "_rects.jpg";
			imwrite(output_filepath, output_image);	
		}
	}
}