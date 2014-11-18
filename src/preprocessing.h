#ifndef NEURONS_PREPROCESING_H
#define NEURONS_PREPROCESING_H

#include <opencv2/opencv.hpp>

#include "ImageRecognition.h"

using namespace std;
using namespace cv;

namespace ImageRecognition
{
	class preprocessing
	{
	public:
		preprocessing( RecognitionStatistics &_stat );
		preprocessing(int _smooth_radius, double _subtr_val, double _thresh_coeff, RecognitionStatistics &_stat);
		preprocessing(const preprocessing& orig, RecognitionStatistics &stat);
		 virtual ~preprocessing();
		void do_prep( const char * , Mat& );
		void do_prep(Mat image, Mat& output_image);
		void do_prep(Mat& output_image);
		void thresholding( Mat& , int thresh = -1 );

		void classifier_prep( const char *, Mat& );
		void classifier_prep( Mat& );

		void copy( preprocessing *);

		static bool check_diff( Mat& , int x, int y, int direction );
	private:
		Mat    main_image;
		int	   smooth_radius;
		double subtr_val;
		double thresh_coeff;
		RecognitionStatistics &stat;
		static const int iter_num = 20;

		static uchar clamp( double value, uchar min, uchar max );

		void load_image( const char * );
		void unsharped_mask( Mat& );
		void close_reconstruction( Mat& );

		void reconstruction( Mat&, Mat& );
		void artems_filter( Mat& );

		double get_contrast();
		static double get_threshold( Mat& );
	};

}
#endif // NEURONS_PREPROCESING_H
