#ifndef SLIDINGWINDOW_H
#define SLIDINGWINDOW_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>

#include "consts.h"

namespace ImageRecognition
{

using namespace std;
using namespace cv;

template<typename TPixel>
struct ResponsePlaceholder
{
    Mat image, grad, grad_x, grad_y;
    Mat angle;

    void Init( Mat& _image )
    {
        image = _image;

        GaussianBlur( image, image, Size(3,3), 0, 0, BORDER_DEFAULT );

        /// Convert it to gray
        //cvtColor( part, part, CV_RGB2GRAY );


        /// Generate grad_x and grad_y
        Mat abs_grad_x, abs_grad_y;

        int scale = 1;
        int delta = 0;

        /// Gradient X
        //Scharr( part, grad_x, CV_16S, 1, 0, scale, delta, BORDER_DEFAULT );
        Sobel( image, grad_x, CV_16S, 1, 0, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_x, abs_grad_x );

        /// Gradient Y
        //Scharr( part, grad_y, CV_16S, 0, 1, scale, delta, BORDER_DEFAULT );
        Sobel( image, grad_y, CV_16S, 0, 1, 3, scale, delta, BORDER_DEFAULT );
        convertScaleAbs( grad_y, abs_grad_y );

        /// Total Gradient (approximate)
        addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad );
        angle = Mat(grad.rows, grad.cols, CV_32FC1, 0.0f);
        for(int y=0; y<grad.rows; ++y)
        {
            for(int x=0; x<grad.cols; ++x)
            {
                //if( abs( grad_x.at<short>(y, x) ) > 1 || abs( grad_y.at<short>(y, x) ) > 1 )
                    angle.at<float>(y, x) = (atan2( double( grad_y.at<short>(y, x) ), double( grad_x.at<short>(y, x) ) ) + 3.1415926)*180.0f/(3.1415926f);
            }
        }

       #if 1
            {
                float mult = 1.0f;
                Mat dir_image( grad.rows, grad.cols, CV_32FC3 );
                Mat dir_image2( grad.rows, grad.cols, CV_32FC1 );
                for(int y=0; y<grad.rows; ++y)
                {
                    for(int x=0; x<grad.cols; ++x)
                    {
                        dir_image.at<Vec3f>(y, x) = Vec3f( 0.0f,
                                                           127.5f + mult*grad_y.at<short>(y, x),
                                                           127.5f + mult*grad_x.at<short>(y, x) );
                        dir_image2.at<float>(y, x) = angle.at<float>(y, x);//*200.0f/360.0f;
                    }
                }
                static int imcount = 1;
                stringstream filename1, filename2, filename3;
                filename1 << "dump/directions" << imcount << ".jpg";
                filename2 << "dump/image" << imcount << ".jpg";
                filename3 << "dump/angle" << imcount << ".jpg";
                ++imcount;
                imwrite( filename1.str(), dir_image );
                imwrite( filename2.str(), image );
                imwrite( filename3.str(), dir_image2 );
            }
        #endif

    }

    float operator()( int pos_x, int pos_y, int width, int height ) const
    {
        #if 0
            Mat part = grad(Range(pos_y, pos_y + height), Range(pos_x, pos_x + width));
            //qDebug() << part.rows << " " << part.cols << endl;
            string filename = "dump/part" + int2str(i) + "_" + int2str(j) + ".png";
            imwrite(filename, part);
        #endif

        //qDebug() << grad.type() << endl;
        double res = 0.0;
        double res2 = 0.0;
        double res3 = 0.0;
        double sum_x = 0.0, sum_y = 0.0;

        const int GRAD_HIST_SIZE = 8;
        vector<int> grad_hist( GRAD_HIST_SIZE + 1 );
        fill(grad_hist.begin(), grad_hist.end(), 0);
        for(int y=0; y<height; ++y)
        {
            for(int x=0; x<width; ++x)
            {
                res += abs( grad.at<uchar>(pos_y + y, pos_x + x) );
                grad_hist[ int( angle.at<float>( pos_y + y, pos_x + x)/360.0f*(GRAD_HIST_SIZE-1) + 0.5f ) ]++;
                sum_x += double( grad_x.at<short>( pos_y + y, pos_x + x) );
                sum_y += double( grad_y.at<short>( pos_y + y, pos_x + x) );
            }
        }
        res3 = pow( abs( sum_x*sum_y*res ), 1.0/3.0 );
        res3 /= width*height;

        res /= width*height;
        res2 = double( grad_hist[0] );
        for(int i=1; i<GRAD_HIST_SIZE; ++i)
            res2 *= double( grad_hist[i] );
        res2 = pow(res2, 1.0/GRAD_HIST_SIZE );
        res2 /= width*height;
        #if 0
            static int image_count = 1;
            stringstream ss1, ss2;
            ss1 << "dump/part_" << image_count << "_" << res <<  ".png";
            ss2 << "dump/part_" << image_count << "_" << res <<  "_.png";
            imwrite( ss1.str(), grad );
            imwrite( ss2.str(), part );
            ++image_count;
        #endif
        return res;
    }
};

class SlidingWindowFragmentation
{
    int standartWindowSize;
    int standartWindowOffset;
    Size standartWindow;
    vector<float> scales;
    vector<unsigned int> sizes;
    float standartWindowSizeToImageSizeRatio;

    Mat baseImage;

    static std::string int2str( int n )
    {
        std::stringstream ss;
        ss << n;
        return ss.str();
    }

public:
    SlidingWindowFragmentation( Mat _image, unsigned int _standartWindowSize,
                                unsigned int _standartWindowOffset, vector<unsigned int> _sizes )
    {
        sizes = _sizes;
        standartWindowSize = _standartWindowSize;
        standartWindowOffset = _standartWindowOffset;

        standartWindow = Size( standartWindowSize, standartWindowSize );

        float ratio1 = standartWindowSize/float(sizes[0]);
        standartWindowSizeToImageSizeRatio = ratio1;

        int h = int(_image.rows*ratio1);
        int w = int(_image.cols*ratio1);
        baseImage = Mat( h, w, _image.type() );
        resize(_image, baseImage, baseImage.size() );

        int sizes_count = sizes.size();
        scales.resize( sizes_count );
        scales[0] = 1.0f;
        for( int i=1; i<sizes_count; ++i )
            scales[i] = sizes[0]/float(sizes[i]);
    }

    template<class ResponseFunctor>
    void FindMaximaWindows( vector< Rect_<int> > &maxima, ResponseFunctor &response)
    {
        struct RectValue
        {
            Rect_<int> rect;
            float value;

            RectValue() {}
            RectValue(Rect_<int> _rect, float _value)
            {
               rect = _rect;
               value = _value;
            }
        };

        vector< RectValue > rects;
        for( int i=0; i<scales.size(); ++i )
        {
            Mat responseImage = MakeResponseImage( i, response );
            Mat responseImageNMS = Mat( responseImage.rows, responseImage.cols, CV_32FC1, 0.0f );

            int delta = (standartWindowSize/2)/standartWindowOffset;
            for( int y=0; y<responseImage.rows; ++y )
            {
                for( int x=0; x<responseImage.cols; ++x )
                {
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
                    //if(max_value < HOGFeatureClassifier::MODEL_THRESHOLD)
                    if(max_value < 0.78f)
                        continue;
                    if( responseImage.at<float>(y, x) == max_value )
                    {
                        responseImageNMS.at<float>(y, x) = responseImage.at<float>(y, x)*100;
                        int pos_x = x*standartWindowOffset/(scales[i]*standartWindowSizeToImageSizeRatio);
                        int pos_y = y*standartWindowOffset/(scales[i]*standartWindowSizeToImageSizeRatio);

                        rects.push_back( RectValue( Rect_<int>( pos_x, pos_y, sizes[i], sizes[i] ), responseImageNMS.at<float>(y, x) ) );
                    }
                }
            }
            #if 1
            {
                stringstream image_filename1, image_filename2;
                image_filename1 << "dump/response" << i << ".png";
                image_filename2 << "dump/responseNMS" << i << ".png";
                imwrite( image_filename1.str(), responseImage );
                imwrite( image_filename2.str(), responseImageNMS );
            }
            #endif
        }
        cout << " done" << endl;
        for(int i=0; i<rects.size(); ++i)
        {
            float square = float( rects[i].rect.width*rects[i].rect.height );
            bool is_max = true;
            for(int j=0; j<rects.size(); ++j)
            {
                if( i == j )
                    continue;
                float currentSquare = min(square, float( rects[j].rect.width*rects[j].rect.height ) );
                if( GetIntersectionRectSquare(rects[i].rect, rects[j].rect)/currentSquare > 0.4 )
                {
                    if( (rects[i].value < rects[j].value) || ( rects[i].value == rects[j].value && i < j) )
                    {
                        is_max = false;
                        break;
                    }
                }
            }
            if(is_max)
                maxima.push_back( rects[i].rect );
        }
    }

    template<class ResponseFunctor>
    Mat MakeResponseImage( unsigned int scale_index, ResponseFunctor &response)
    {
        if(scale_index >= scales.size())
            throw "Index Out of Range";

        int w = int(baseImage.cols*scales[ scale_index ]);
        int h = int(baseImage.rows*scales[ scale_index ]);
        Mat image = Mat( h, w, baseImage.type() );
        resize( baseImage, image, image.size() );
        imwrite("dump/resizedImage_" + int2str(scale_index) + ".jpg", image);

        response.Init( image );

        int response_width = w/standartWindowOffset - standartWindowSize/standartWindowOffset - 1;
        int response_height = h/standartWindowOffset - standartWindowSize/standartWindowOffset - 1;
        Mat responseImage = Mat( response_height, response_width, CV_32FC1 );
        float maxResponse = 0.0f;

        for( int i=0; i<responseImage.rows; ++i )
        {
            if((i+1)%100 == 0)
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
        cout << "*" << endl;
        cout << "maxResponse: " << maxResponse << endl;
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
private:

    static int GetIntersectionRectSquare( const Rect_<int> &rect1, const Rect_<int> &rect2 )
    {
        int minX = max( rect1.x, rect2.x );
        int minY = max( rect1.y, rect2.y );

        int maxX = min( rect1.x + rect1.width, rect2.x + rect2.width );
        int maxY = min( rect1.y + rect1.height, rect2.y + rect2.height );

        int width = max( maxX - minX, 0 );
        int height = max( maxY - minY, 0 );

        return width*height;
    }
};

}

#endif // SLIDINGWINDOW_H
