#ifndef SLIDINGWINDOW_H
#define SLIDINGWINDOW_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>

#include "ImageRecognition.h"
#include "consts.h"
#include "utils.h"

namespace ImageRecognition
{

using namespace std;
using namespace cv;

class SlidingWindowFragmentation
{
    int standartWindowSize;
    int standartWindowOffset;
    Size standartWindow;
    vector<float> scales;
    vector<int> sizes;
    float standartWindowSizeToImageSizeRatio;

    Mat baseImage, basePrepImage;
	RecognitionStatistics &stat;
	const HOGContext &context;
public:
	SlidingWindowFragmentation(Mat _image, Mat _prepImage, const HOGContext &_context, RecognitionStatistics &_stat) :
		stat(_stat), context(_context)
    {
        sizes = context.slidingWindowSizes;
		standartWindowSize = context.standartSlidingWindowSize;
		standartWindowOffset = context.standartslidingWindowStep;

        standartWindow = Size( standartWindowSize, standartWindowSize );

        float ratio1 = standartWindowSize/float(sizes[0]);
        standartWindowSizeToImageSizeRatio = ratio1;

        int h = int(_image.rows*ratio1);
        int w = int(_image.cols*ratio1);
        baseImage = Mat( h, w, _image.type() );
        resize(_image, baseImage, baseImage.size() );

		basePrepImage = Mat(h, w, CV_8U);
		resize(_prepImage, basePrepImage, basePrepImage.size(), 0.0, 0.0, CV_INTER_NN);

        int sizes_count = sizes.size();
        scales.resize( sizes_count );
        scales[0] = 1.0f;
        for( int i=1; i<sizes_count; ++i )
            scales[i] = sizes[0]/float(sizes[i]);
    }

    template<class ResponseFunctor>
	void FindMaximaWindows(vector< SlidingRect > &maxima, ResponseFunctor &response)
    {
		clock_t begin_time = clock();

        vector< SlidingRect > rects;
        for( int i=0; i<scales.size(); ++i )
        {
            Mat responseImage = MakeResponseImage( i, response );
            Mat responseImageNMS = Mat( responseImage.rows, responseImage.cols, CV_32FC1, 0.0f );

            for( int y=0; y<responseImage.rows; ++y )
            {
                for( int x=0; x<responseImage.cols; ++x )
                {
					if (responseImage.at<float>(y, x) < 0.00001f)
						continue;
			
					//int delta = (standartWindowSize/2)/standartWindowOffset;
					int delta = 2;
                    float max_value = 0.0f;
                    for(int dy=-delta; dy<=delta; ++dy)
                    {
                        for(int dx=-delta; dx<=delta; ++dx)
                        {
                            //if(dx == 0 || dy == 0)
                            //    continue;

                            int _x = x+dx;
                            int _y = y+dy;
                            if( _x < 0 || _x >= responseImage.cols || _y < 0 || _y >= responseImage.rows )
                                continue;

                            max_value = max( max_value, responseImage.at<float>(_y, _x) );
                        }
                    }

                    if( responseImage.at<float>(y, x) == max_value )
                    {
                        responseImageNMS.at<float>(y, x) = responseImage.at<float>(y, x);
                        int pos_x = x*standartWindowOffset/(scales[i]*standartWindowSizeToImageSizeRatio);
                        int pos_y = y*standartWindowOffset/(scales[i]*standartWindowSizeToImageSizeRatio);

                        rects.push_back( SlidingRect( Rect_<int>( pos_x, pos_y, sizes[i], sizes[i] ), responseImageNMS.at<float>(y, x) ) );
                    }
                }
            }
            #if 0
            {
                stringstream image_filename1, image_filename2;
                image_filename1 << "dump/response" << i << ".png";
                image_filename2 << "dump/responseNMS" << i << ".png";
                imwrite( image_filename1.str(), responseImage );
                imwrite( image_filename2.str(), responseImageNMS );
            }
            #endif
        }
		clock_t end_time = clock();
		if (stat.flOutputTime && false)
			cout << "Sliding Window Time: " << (float(end_time - begin_time) / 1000.0f) << endl;
		using Utils::sqr;
        for(int i=0; i<rects.size(); ++i)
        {
			float x = rects[i].rect.x + float(rects[i].rect.width) / 2.0f;
			float y = rects[i].rect.y + float(rects[i].rect.height) / 2.0f;
			float w = rects[i].rect.width;

            bool is_max = true;
            for(int j=0; j<rects.size(); ++j)
            {
                if( i == j )
                    continue;

				float _x = rects[j].rect.x + float(rects[j].rect.width) / 2.0f;
				float _y = rects[j].rect.y + float(rects[j].rect.height) / 2.0f;
				float _w = max( w, float( rects[j].rect.width ) );
				float distanse = sqrtf(sqr(x - _x) + sqr(y - _y));

				if (distanse < 0.4f*_w)
                {
                    if( (rects[i].value < rects[j].value) || ( rects[i].value == rects[j].value && i < j) )
                    {
                        is_max = false;
                        break;
                    }
                }
            }
            if(is_max)
                maxima.push_back( rects[i] );
        }
    }

    template<class ResponseFunctor>
    Mat MakeResponseImage( unsigned int scale_index, ResponseFunctor &response)
    {
		using Utils::int2str;
        if(scale_index >= scales.size())
            throw "Index Out of Range";

        int w = int(baseImage.cols*scales[ scale_index ]);
        int h = int(baseImage.rows*scales[ scale_index ]);
        Mat image = Mat( h, w, baseImage.type() );
        resize( baseImage, image, image.size() );
		
		Mat prepImage = Mat(h, w, CV_8U);
		resize(basePrepImage, prepImage, prepImage.size(), 0.0, 0.0, CV_INTER_NN);
		
		if (stat.flDumpDebugImages && false)
		{
			imwrite("dump/resizedImage_" + int2str(scale_index) + ".jpg", image);
			Mat tmp = prepImage * 255;
			bitwise_not(tmp, tmp);
			imwrite("dump/resizedPrepImage_" + int2str(scale_index) + ".jpg", tmp);
		}

		clock_t begin_time = clock();
		if (stat.flOutputInfo && false)
			std::cout << "Response image";

        response.Init( image, prepImage );

        int response_width = w/standartWindowOffset - standartWindowSize/standartWindowOffset - 1;
        int response_height = h/standartWindowOffset - standartWindowSize/standartWindowOffset - 1;
        Mat responseImage = Mat( response_height, response_width, CV_32FC1 );
        float maxResponse = 0.0f;

        for( int i=0; i<responseImage.rows; ++i )
        {
			if (stat.flOutputInfo && false)
            if((i+1)%10 == 0)
                cout << ".";

            int y = i*standartWindowOffset;

            for( int j=0; j<responseImage.cols; ++j )
            {
                int x = j*standartWindowOffset;
                responseImage.at<float>(i, j) = max( response( x, y, standartWindowSize, standartWindowSize), 0.0f );
                //cout << responseImage.at<float>(i, j) << endl;
                maxResponse = max(maxResponse, responseImage.at<float>(i, j));
            }
        }
		if (stat.flOutputInfo && false)
		{
			cout << "*" << endl;
			//cout << "maxResponse: " << maxResponse << endl;
			//cout << "debug count operator: " << response.DEBUG_COUNT_OPERATOR << endl;
		}
		if (stat.flOutputTime && false)
		{
			clock_t end_time = clock();
			cout << "Response Image Time: " << (float(end_time - begin_time) / 1000.0f) << endl;
		}
        //responseImage /= maxResponse;
        //responseImage *= 255;
        /*
        int border = (standartWindowOffset/2)/standartWindowOffset;
        Mat responseBorderedImage;
        copyMakeBorder( responseImage, responseBorderedImage,
                        border, border, border, border,
                        BORDER_CONSTANT, Scalar ( 0, 0, 0 ) );

        return responseBorderedImage;
        */
        return responseImage;
    }
};

}

#endif // SLIDINGWINDOW_H
