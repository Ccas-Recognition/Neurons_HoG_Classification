#include "preprocessing.h"
#include "consts.h"
#include "utils.h"
#include <cmath>
namespace ImageRecognition
{
	preprocessing::preprocessing(RecognitionStatistics &_stat)
		: smooth_radius(31), subtr_val(3.2), thresh_coeff(0.65), stat(_stat)
	{
	}

	preprocessing::preprocessing(int _smooth_radius, double _subtr_val, double _thresh_coeff, RecognitionStatistics &_stat)
		: smooth_radius(_smooth_radius), subtr_val(_subtr_val), thresh_coeff(_thresh_coeff), stat(_stat)
	{
	}

	preprocessing::preprocessing(const preprocessing& orig, RecognitionStatistics &_stat):
		stat(_stat)
	{
		if (main_image.data) {
			main_image = orig.main_image.clone();
		}
		subtr_val = orig.subtr_val;
		smooth_radius = orig.smooth_radius;
		thresh_coeff = orig.thresh_coeff;
	}

	void preprocessing::copy(preprocessing *orig)
	{
		orig->main_image = main_image.clone();
		orig->subtr_val = subtr_val;
		orig->smooth_radius = smooth_radius;
		orig->thresh_coeff = thresh_coeff;
	}

	preprocessing :: ~preprocessing()
	{
		if (main_image.data) {
			main_image.release();
		}
	}

	void preprocessing::load_image(const char *image_name)
	{
		if (main_image.data) {
			main_image.release();
		}

		main_image = imread(image_name, 0);
	}

	void preprocessing::do_prep(const char *image_name, Mat& output_image)
	{
		load_image(image_name);
		do_prep(output_image);
	}
	void preprocessing::do_prep(Mat image, Mat& output_image)
	{
		main_image = image;
		do_prep(output_image);
	}
	void preprocessing::do_prep(Mat& output_image)
	{
		unsharped_mask(output_image);
		close_reconstruction(output_image);
		//imwrite("proc_image1.bmp", output_image);
		thresholding(output_image);
		//imwrite("proc_image2.bmp", output_image);
		Mat se = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
		erode(output_image, output_image, se);
		//imwrite("proc_image3.bmp", output_image);
		dilate(output_image, output_image, se);
		//imwrite("proc_image4.bmp", output_image);


		for (int y = 0; y < output_image.rows; ++y)
		{
			for (int x = 0; x < output_image.cols; ++x)
			{
				output_image.at<uchar>(y, x) = (output_image.at<uchar>(y, x) == 0) ? 1 : 0;
			}
		}
		if (stat.flDumpDebugImages)
		{
			static int index = 1;
			imwrite("dump/image"+ Utils::int2str(index++) +".bmp", main_image);
			Mat tmp = output_image * 255;
			bitwise_not(tmp, tmp);
			imwrite("dump/bw" + Utils::int2str(index++) + ".bmp", tmp);
		}
	}


	bool preprocessing::check_diff(Mat &proc_image, int x, int y, int direction)
	{
		int new_x = x, new_y = y;

		switch (direction) {
		case 1: --new_x; break;
		case 2: ++new_x; break;
		case 3: --new_y; break;
		case 4: ++new_y; break;
		}

		if (proc_image.ptr(y)[x] == proc_image.ptr(new_y)[new_x])
			return true;
		else
			return false;
	}

	/*
	vector< cluster >& preprocessing::get_conn(const char *image_name, Mat& output_image, bool prepr, bool post)
	{
		srand(282827);

		if (prepr && post) {
			do_prep(image_name, output_image);
			//imwrite( "D:\\Neurons\\bw.bmp", output_image );
		}

		int width = output_image.cols,
			height = output_image.rows;
		vector< cluster > *clusters_data = new vector< cluster >();

		unsigned int cluster_size = 0, tmp_color_sum = 0;
		int cluster_num = 0, _x = 0, _y = 0;
		cluster query_vec;
		uchar *image_data = NULL;


		int **cluster_mat = new int*[width];
		for (int i = 0; i < width; ++i)
			cluster_mat[i] = new int[height];

		for (int i = 0; i < width; ++i)
		for (int j = 0; j < height; ++j)
			cluster_mat[i][j] = -1;

		for (int j = 0; j < height; ++j) {
			image_data = output_image.ptr(j);
			for (int i = 0; i < width; ++i) {

				if (cluster_mat[i][j] == -1) {

					++cluster_num;
					query_vec.set_point(i, j);
					cluster_mat[i][j] = cluster_num;
					++cluster_size;
					tmp_color_sum += (unsigned int)image_data[i];

					for (unsigned int k = 0; k < query_vec.length(); ++k) {
						_x = query_vec[k].x();
						_y = query_vec[k].y();

						if ((_x - 1 >= 0) && (cluster_mat[_x - 1][_y] == -1) && check_diff(output_image, _x, _y, 1)) {
							cluster_mat[_x - 1][_y] = cluster_num;
							query_vec.set_point(_x - 1, _y);
							++cluster_size;
							tmp_color_sum += (unsigned int)output_image.ptr(_y)[_x - 1];
						}

						if ((_x + 1 < width) && (cluster_mat[_x + 1][_y] == -1) && check_diff(output_image, _x, _y, 2)) {
							cluster_mat[_x + 1][_y] = cluster_num;
							query_vec.set_point(_x + 1, _y);
							++cluster_size;
							tmp_color_sum += (unsigned int)output_image.ptr(_y)[_x + 1];
						}

						if ((_y - 1 >= 0) && (cluster_mat[_x][_y - 1] == -1) && check_diff(output_image, _x, _y, 3)) {
							cluster_mat[_x][_y - 1] = cluster_num;
							query_vec.set_point(_x, _y - 1);
							++cluster_size;
							tmp_color_sum += (unsigned int)output_image.ptr(_y - 1)[_x];
						}

						if ((_y + 1 < height) && (cluster_mat[_x][_y + 1] == -1) && check_diff(output_image, _x, _y, 4)) {
							cluster_mat[_x][_y + 1] = cluster_num;
							query_vec.set_point(_x, _y + 1);
							++cluster_size;
							tmp_color_sum += (unsigned int)output_image.ptr(_y + 1)[_x];
						}
					}

					if (cluster_size > 60 && tmp_color_sum == 0.0 && prepr) {
						clusters_data->push_back(query_vec);
					}
					else if (prepr == false && tmp_color_sum > 0.0 && cluster_size > 7) {
						clusters_data->push_back(query_vec);
					}

					query_vec.clear();
					cluster_size = 0;
					tmp_color_sum = 0;
				}
			}
		}

		for (int i = 0; i < width; ++i)
			delete[] cluster_mat[i];

		delete[] cluster_mat;

		return *clusters_data;
	}
	*/

	uchar preprocessing::clamp(double value, uchar min, uchar max)
	{
		if (value < min)
			return min;
		else
		if (value > max)
			return max;
		else
			return (uchar)value;
	}

	void preprocessing::unsharped_mask(Mat& proc_image)
	{
		Mat smooth_image;
		int width = main_image.cols,
			height = main_image.rows;
		proc_image = Mat::zeros(main_image.rows, main_image.cols, main_image.type());

		GaussianBlur(main_image, smooth_image, Size(smooth_radius, smooth_radius), 2.5, 1.5);

		double tmp_val = 0.0;

		for (int j = 0; j < height; ++j) {
			uchar *main_image_data = main_image.ptr(j);
			uchar *smooth_image_data = smooth_image.ptr(j);
			uchar *proc_image_data = proc_image.ptr(j);

			for (int i = 0; i < width; ++i) {
				tmp_val = main_image_data[i] * 1.0 +
					subtr_val * (main_image_data[i] * 1.0 - smooth_image_data[i] * 1.0);

				proc_image_data[i] = clamp(tmp_val, 0, 255);
			}
		}

		smooth_image.release();
	}

	void preprocessing::close_reconstruction(Mat& proc_image)
	{

		int width = proc_image.cols,
			height = proc_image.rows;
		Mat se = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 4)),
			conn = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)),
			cur_image = proc_image.clone();
		uchar *cur_row_main = NULL;
		uchar *cur_row_dist = NULL;

		dilate(proc_image, proc_image, se);

		for (int k = 0; k < iter_num; ++k) {
			erode(proc_image, proc_image, conn);

			for (int j = 0; j < height; ++j) {
				cur_row_dist = proc_image.ptr(j);
				cur_row_main = cur_image.ptr(j);

				for (int i = 0; i < width; ++i) {
					cur_row_dist[i] = (cur_row_main[i] > cur_row_dist[i]) ?
						cur_row_main[i] : cur_row_dist[i];
				}
			}
		}

		cur_image.release();
	}

	double preprocessing::get_threshold(Mat& proc_image)
	{
		int w = proc_image.cols;
		int h = proc_image.rows;

		int max = 0;
		int min = 255;
		double T = 0.0;

		bool done = false;
		T = 0.5 * (min * 1.0 + max * 1.0);

		double Tnext = 0.0;

		while (done == false) {

			double mean = 0.0;
			int num_mean = 0;
			double ant_mean = 0.0;
			int num_ant_mean = 0;

			for (int j = 0; j < h; ++j) {
				uchar *proc_image_col = proc_image.ptr(j);
				for (int i = 0; i < w; ++i) {

					if ((double)proc_image_col[i] > T) {
						mean += (double)proc_image_col[i];
						++num_mean;
					}
					else {
						ant_mean += (double)proc_image_col[i];
						++num_ant_mean;
					}
				}
			}

			mean = (num_mean == 0) ? T : mean / (num_mean * 1.0);
			ant_mean = (num_ant_mean == 0) ? T : ant_mean / (num_ant_mean * 1.0);

			Tnext = 0.5 * (mean + ant_mean);

			done = (fabs(T - Tnext) < 0.5);
			T = Tnext;
		}

		return T;
	}

	double preprocessing::get_contrast()
	{
		uchar *main_image_data = NULL;
		long int pixels_sum = 0;
		double mean_val = 0.0;
		int width = main_image.cols,
			height = main_image.rows;


		for (int j = 0; j < height; ++j) {
			main_image_data = (uchar *)main_image.ptr(j);
			for (int i = 0; i < width; ++i) {
				pixels_sum += (long int)main_image_data[i];
			}
		}
		mean_val = (pixels_sum * 1.0) / (width * height * 1.0);

		double tmp_sum = 0.0;

		for (int j = 0; j < height; ++j) {
			main_image_data = (uchar *)main_image.ptr(j);
			for (int i = 0; i < width; ++i) {
				tmp_sum += (main_image_data[i] * 1.0 - mean_val) * (main_image_data[i] * 1.0 - mean_val);
			}
		}
		return sqrt(tmp_sum / (width * height * 1.0)) / 255.0;
	}

	void preprocessing::thresholding(Mat& proc_image, int thresh)
	{
		if (thresh == -1) {
			double T = get_threshold(proc_image);
			double contrast = get_contrast();
			thresh_coeff = 0.6315;//contrast * 10 + ( 0.86 - contrast * 10.0 );
			//adaptiveThreshold( proc_image, proc_image, 255.0, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 79, 0.0 );
			//imwrite( "thresh_adapt_image.bmp", proc_image );
			threshold(proc_image, proc_image, T * thresh_coeff, 255, THRESH_BINARY);
		}
		else {
			threshold(proc_image, proc_image, thresh * 1.0, 255, THRESH_BINARY);
		}
	}

	void preprocessing::reconstruction(Mat& mask, Mat& marker)
	{
		Mat conn = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		int width = mask.cols,
			height = mask.rows;
		uchar *cur_row_mask = NULL;
		uchar *cur_row_marker = NULL;

		for (int k = 0; k < iter_num; ++k) {
			dilate(marker, marker, conn);

			for (int j = 0; j < height; ++j) {
				cur_row_mask = mask.ptr(j);
				cur_row_marker = marker.ptr(j);

				for (int i = 0; i < width; ++i) {
					cur_row_marker[i] = (cur_row_mask[i] < cur_row_marker[i]) ?
						cur_row_mask[i] : cur_row_marker[i];
				}
			}
		}
	}

	void preprocessing::artems_filter(Mat& proc_image)
	{
		Mat first_image;
		Mat se = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
		morphologyEx(main_image, first_image, cv::MORPH_OPEN, se);
		reconstruction(main_image, first_image);

		Mat neg_image,
			second_image,
			white_image = Mat(main_image.rows, main_image.cols, main_image.type(), 255);
		subtract(white_image, first_image, neg_image);
		se = getStructuringElement(MORPH_ELLIPSE, Size(21, 21));
		morphologyEx(neg_image, second_image, cv::MORPH_OPEN, se);
		reconstruction(neg_image, second_image);

		Mat thirdth_neg_image;
		subtract(white_image, second_image, thirdth_neg_image);
		subtract(thirdth_neg_image, first_image, thirdth_neg_image);

		se = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
		morphologyEx(thirdth_neg_image, proc_image, cv::MORPH_OPEN, se);
		reconstruction(thirdth_neg_image, proc_image);
		subtract(white_image, proc_image, proc_image);
	}

	void preprocessing::classifier_prep(const char *image_name, Mat& output_image)
	{
		load_image(image_name);
		artems_filter(output_image);
	}

	void preprocessing::classifier_prep(Mat& output_image)
	{
		artems_filter(output_image);
	}
}//namespace ImageRecognition