#include "SlidingWindowOptimizataion.h"
#include "HOG.h"
#include "preprocessing.h"
#include "utils.h"

#include <sstream>
#include <time.h>
#include <opencv2/opencv.hpp>

using HOGFeatureClassifier::TModel;
using namespace cv;

namespace ImageRecognition
{
	void ResponseImage(vector<ImageRecognition::SlidingRect> &rects, const Mat &image, const TModel &model, RecognitionStatistics &stat)
	{
		GetRectsFromImage(rects, image, model, stat);
		for (int i = 0; i < rects.size(); ++i)
			rects[i].falseDetection = (rects[i].value < model.getModelThreshold());
	}

	void ResponseImage(vector<ImageRecognition::SlidingRect> &rects, const Mat &image, const string &model_filename, RecognitionStatistics &stat)
	{
		TModel model;
		model.Load(model_filename);
		ResponseImage(rects, image, model, stat);
	}

	void GetRectsFromImage(vector<ImageRecognition::SlidingRect> &rects, const Mat &image, const TModel &model, RecognitionStatistics &stat)
	{
		using namespace ImageRecognition;
		vector< unsigned int > sizes(sizeof(sizes_) / sizeof(unsigned int));
		copy(sizes_, sizes_ + sizeof(sizes_) / sizeof(unsigned int), sizes.begin());

		ImageRecognition::preprocessing prep(stat);
		Mat prepImage;

		clock_t begin_time = clock();

		prep.do_prep(image, prepImage);
		//if (stat.flOutputTime && false)
		//	cout << "Preprocessing Time: " << (float(end_time - begin_time) / 1000.0f) << endl;

		ImageRecognition::SlidingWindowFragmentation fragmentation(image, prepImage, model.GetContext(), stat);
		HOGFeatureClassifier::HoGResponseFunctor response(stat);
		response.InitModel(&model);
		fragmentation.FindMaximaWindows(rects, response);

		clock_t end_time = clock();
		if (stat.flOutputTime)
			cout << "Recognition Time: " << (float(end_time - begin_time) / 1000.0f) << endl;

	}

	void ReadRightRects(vector<ImageRecognition::SlidingRect> &rightRects, const string &xml_filename, RecognitionStatistics &stat)
	{
		using namespace Utils;

		FileStorage file_storage(xml_filename, FileStorage::READ);
		FileNode images = file_storage["images"];
		rightRects.reserve(images.size());

		for (FileNodeIterator it = images.begin(); it != images.end(); ++it)
		{
			string part_filename = string(*it);
			int dot_pos = part_filename.find_first_of('.');
			if (dot_pos != -1)
				part_filename = part_filename.substr(0, dot_pos);

			stringstream ss(part_filename);
			vector<string> parts;
			string part;
			while (getline(ss, part, '_'))
				parts.push_back(part);

			rightRects.push_back(ImageRecognition::SlidingRect());
			int last = parts.size() - 1;
			rightRects.back().rect.x = str2int(parts[last - 3]);
			rightRects.back().rect.y = str2int(parts[last - 2]);
			rightRects.back().rect.width = str2int(parts[last - 1]);
			rightRects.back().rect.height = str2int(parts[last]);

		}
	}

	void GetCheckingRects(vector<ImageRecognition::SlidingRect> &rects, vector<ImageRecognition::SlidingRect> &rightRects, const Mat &image, const string& xml_filename, const TModel &model, RecognitionStatistics &stat)
	{
		ReadRightRects(rightRects, xml_filename, stat);
		GetRectsFromImage(rects, image, model, stat);
		using Utils::sqr;

		for (int i = 0; i < rects.size(); ++i)
		{
			rects[i].falseDetection = true;

			float x = rects[i].rect.x + float(rects[i].rect.width) / 2.0f;
			float y = rects[i].rect.y + float(rects[i].rect.height) / 2.0f;
			float w = rects[i].rect.width;

			for (int j = 0; j < rightRects.size(); ++j)
			{
				float _x = rightRects[j].rect.x + float(rightRects[j].rect.width) / 2.0f;
				float _y = rightRects[j].rect.y + float(rightRects[j].rect.height) / 2.0f;
				float _w = rightRects[j].rect.width;

				float distanse = sqrtf(sqr(x - _x) + sqr(y - _y));
				if (distanse < 0.2f*_w && min(w, _w) / max(w, _w) > 0.6f)
				{
					rightRects[j].falseDetection = true;
					rects[i].falseDetection = false;
					break;
				}
			}
		}
	}

	float FindOptimalThresholdForModel(const vector<ImageRecognition::SlidingRect> &rects, int &balanced_mis_count, int &false_positive_count,
		int right_rects_count, int absolute_mis_count, RecognitionStatistics &stat)
	{
		float max_value = 0.0f;
		for (auto it = rects.begin(); it != rects.end(); ++it)
		{
			max_value = max(max_value, it->value);
		}
		//cout << "Max Value: " << max_value << endl;
		float min_error = 10000;
		float min_threshold = 0.0f;

		//const float k1 = 0.20f;
		//const float k2 = 0.80f;
		const float k1 = 1.0f;
		const float k2 = 1.0f;

		const int GRID_COUNT = 100;
		stat.recognitionMissigs.resize(GRID_COUNT);
		stat.recognitionFalseDetections.resize(GRID_COUNT);
		for (int k = 0; k < 100; ++k)
		{
			float t = k / float(GRID_COUNT);

			float current_threshold = k / float(GRID_COUNT)*max_value;
			int current_mis_count = 0;
			int current_false_positive_count = 0;

			for (auto it = rects.begin(); it != rects.end(); ++it)
			{
				if (it->value > current_threshold)
				{
					if (it->falseDetection)
						++current_false_positive_count;
				}
				else
				{
					if (!it->falseDetection)
						++current_mis_count;
				}
			}
			float current_error = k1*current_false_positive_count + k2*(current_mis_count + absolute_mis_count);
			if (min_error > current_error)
			{
				min_error = current_error;
				min_threshold = current_threshold;
				balanced_mis_count = current_mis_count;
				false_positive_count = current_false_positive_count;
				stat.recognitionMinMissings = (current_mis_count + absolute_mis_count) / float(right_rects_count);
				stat.recognitionMinFalseDetections = current_false_positive_count / float(right_rects_count);
			}
			stat.recognitionMissigs[k].value = t;
			stat.recognitionMissigs[k].error = (current_mis_count + absolute_mis_count) / float(right_rects_count);
			stat.recognitionFalseDetections[k].value = t;
			stat.recognitionFalseDetections[k].error = current_false_positive_count / float(right_rects_count);
		}
		return min_threshold;
	}

	float FindOptimalThresholdForModel(const string &images_list, const TModel &model, RecognitionStatistics &stat)
	{
		vector<ImageRecognition::SlidingRect> rects;
		vector< vector<ImageRecognition::SlidingRect> > check_rects;
		vector< vector<ImageRecognition::SlidingRect> > check_right_rects;
		vector< string > check_output_images;

		string dir = "";
		ifstream images(images_list);
		if (!images.is_open())
			return 0.0f;
		int dir_index = images_list.find_last_of('/');
		if (dir_index == -1)
			images_list.find_last_of('\\');
		if (dir_index != -1)
			dir = images_list.substr(0, dir_index + 1);

		int mis_count = 0;
		int right_rects_count = 0;

		while (!images.eof())
		{
			string xml_filename;
			images >> xml_filename;
			std::stringstream ss(xml_filename);
			std::string item;

			string filename;
			while (getline(ss, item, '_')) {
				if (item[0] == 'r' && item[1] == 'n' && item[2] == 'd')
					break;
				if (filename != "")
					filename = filename + "_";
				filename = filename + item;
			}
			filename += ".jpg";
			filename = dir + filename;
			xml_filename = dir + xml_filename;

			Mat image = imread(filename, 0);
			if (stat.flOutputInfo)
				cout << filename << " loaded" << endl;
			if (image.rows == 0)
				continue;

			vector<ImageRecognition::SlidingRect> rightRectsForImage;
			vector<ImageRecognition::SlidingRect> rectsForImage;
			GetCheckingRects(rectsForImage, rightRectsForImage, image, xml_filename, model, stat);
			for (auto it = rectsForImage.begin(); it != rectsForImage.end(); ++it)
				rects.push_back(*it);

			right_rects_count += rightRectsForImage.size();
			for (auto it = rightRectsForImage.begin(); it != rightRectsForImage.end(); ++it)
			if (it->falseDetection == false)
				++mis_count;

			if (stat.flDumpDebugImages)
			{
				check_rects.push_back(rectsForImage);
				check_right_rects.push_back(rightRectsForImage);
				check_output_images.push_back(filename);
			}
		}
		int balanced_mis_count = 0;
		int false_positive_count = 0;

		float threshold = FindOptimalThresholdForModel(rects, balanced_mis_count, false_positive_count, right_rects_count, mis_count, stat);
		//threshold = 1.0f;
		/*
		if (stat.flOutputInfo)
		{
			float missings_optimize = float(balanced_mis_count) / right_rects_count * 100.0f;
			float missings = float(balanced_mis_count + mis_count) / right_rects_count * 100.0f;
			float false_positives = float(false_positive_count) / right_rects_count * 100.0f;
			cout << "Missings: " << missings << endl;
			cout << "Missings Optimized: " << missings_optimize << endl;
			cout << "False Positives: " << false_positives << endl;
			cout << "Threshold: " << threshold << endl;
		}
		*/
		if (stat.flDumpDebugImages)
		{
			using Utils::int2str;
			for (int i = 0; i < check_output_images.size(); ++i)
			{
				Mat _output_image = imread(check_output_images[i], 0);
				Mat output_image(_output_image.rows, _output_image.cols, CV_8UC3);
				Mat output_image_positive(_output_image.rows, _output_image.cols, CV_8UC3);
				cvtColor(_output_image, output_image, CV_GRAY2BGR);
				cvtColor(_output_image, output_image_positive, CV_GRAY2BGR);

				for (int j = 0; j < check_rects[i].size(); ++j)
				{
					const auto& current_rect = check_rects[i][j];
					//cout << current_rect.value << endl;
					if (current_rect.value > threshold)
					{
						if (current_rect.falseDetection)
						{
							rectangle(output_image, current_rect.rect, Scalar(0, 0, 255), 1);
							rectangle(output_image_positive, current_rect.rect, Scalar(0, 255, 255), 1);
						}
						else
						{
							rectangle(output_image, current_rect.rect, Scalar(0, 255, 0), 1);
							rectangle(output_image_positive, current_rect.rect, Scalar(0, 255, 0), 1);
						}
					}
					else
					{
						if (!current_rect.falseDetection)
							rectangle(output_image, current_rect.rect, Scalar(255, 0, 0), 1);
					}
				}
				for (int j = 0; j < check_right_rects[i].size(); ++j)
				{
					const auto& current_rect = check_right_rects[i][j];
					if (!current_rect.falseDetection)
						rectangle(output_image, current_rect.rect, Scalar(255, 127, 0), 1);
				}

				imwrite("dump/rects" + int2str(i + 1) + ".png", output_image);
				imwrite("dump/rects_positive" + int2str(i + 1) + ".png", output_image_positive);
			}
		}

		return threshold;
	}
}