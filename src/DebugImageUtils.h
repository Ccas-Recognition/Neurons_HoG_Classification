#ifndef HOG_FEATURES_AND_CLASSIFIER_MMP_DEBUG_IMAGE_UTILS
#define HOG_FEATURES_AND_CLASSIFIER_MMP_DEBUG_IMAGE_UTILS

#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace Utils
{
	struct ImageClusterDumping
	{
		string path;
		int partWidth;
		int partHeight;
		int gridCountX;
		int gridCountY;
		Mat currentCluster;
		int currentImageIndex;
		int currentClusterIndex;

		ImageClusterDumping(string _path, int _partWidth, int _partHeight, int _gridCountX, int _gridCountY )
		{
			path = _path;
			partWidth = _partWidth;
			partHeight = _partHeight;
			gridCountX = _gridCountX;
			gridCountY = _gridCountY;
			
			currentCluster = Mat(_partHeight*_gridCountY, _partWidth*_gridCountX, CV_8UC1, Scalar(0));
			currentImageIndex = 0;
			currentClusterIndex = 1;
		}

		void Pushimage(const Mat &image)
		{
			int imageIndexX = currentImageIndex % gridCountX;
			int imageIndexY = currentImageIndex / gridCountX;
			int posX = imageIndexX*partWidth;
			int posY = imageIndexY*partHeight;

			for (int y = 0; y < partHeight; ++y)
			{
				for (int x = 0; x < partWidth; ++x)
				{
					currentCluster.at<uchar>(y + posY, x + posX) = image.at<uchar>(y, x);
				}
			}
			++currentImageIndex;
			if (currentImageIndex == gridCountX*gridCountY)
			{
				stringstream ss;
				ss << path << currentClusterIndex << ".png";
				imwrite(ss.str(), currentCluster);
				++currentClusterIndex;
				currentImageIndex = 0;
			}
		}

		void DumpOthers()
		{
			if (currentImageIndex == 0)
				return;
			stringstream ss;
			ss << path << currentClusterIndex << ".png";
			imwrite(ss.str(), currentCluster);
			++currentClusterIndex;
			currentImageIndex = 0;
		}
	};

	struct ImageWithNeighborsClusterDumping
	{
		string path;
		int partWidth;
		int partHeight;
		int gridCountX;
		int gridCountY;
		int neighborsCount;
		Mat currentCluster;
		int currentImageIndex;
		int currentClusterIndex;

		ImageWithNeighborsClusterDumping(string _path, int _partWidth, int _partHeight, int _gridCountX, int _gridCountY, int _neighborsCount)
		{
			path = _path;
			partWidth = _partWidth;
			partHeight = _partHeight;
			gridCountX = _gridCountX;
			gridCountY = _gridCountY;

			neighborsCount = _neighborsCount;
			currentCluster = Mat(_partHeight*_gridCountY*(neighborsCount + 2), _partWidth*_gridCountX, CV_8UC3, Scalar(0));
			currentImageIndex = 0;
			currentClusterIndex = 1;
		}

		void Pushimage(const Mat &image, const vector<const Mat*> &neighbors, const vector<int> &neighborsLabels)
		{
			int imageIndexX = currentImageIndex % gridCountX;
			int imageIndexY = currentImageIndex / gridCountX;
			int posX = imageIndexX*partWidth;
			int posY = imageIndexY*partHeight*(neighborsCount + 2);
			
			for (int y = 0; y < partHeight; ++y)
			for (int x = 0; x < partWidth; ++x)
				currentCluster.at<Vec3b>(y + posY, x + posX) = Vec3b(image.at<uchar>(y, x), image.at<uchar>(y, x), image.at<uchar>(y, x));
			for (int i = 0; i < neighbors.size(); ++i)
			{
				posY += partHeight;

				for (int y = 0; y < partHeight; ++y)
				for (int x = 0; x < partWidth; ++x)
					currentCluster.at<Vec3b>(y + posY, x + posX) = Vec3b(neighbors[i]->at<uchar>(y, x), neighbors[i]->at<uchar>(y, x), neighbors[i]->at<uchar>(y, x));
				currentCluster.at<Vec3b>(posY, posX) = (neighborsLabels[i] == 1) ? Vec3b(0, 255, 0) : Vec3b(0, 0, 255);
			}

			++currentImageIndex;
			if (currentImageIndex == gridCountX*gridCountY)
			{
				stringstream ss;
				ss << path << currentClusterIndex << ".png";
				imwrite(ss.str(), currentCluster);
				++currentClusterIndex;
				currentImageIndex = 0;
			}
		}

		void DumpOthers()
		{
			if (currentImageIndex == 0)
				return;
			stringstream ss;
			ss << path << currentClusterIndex << ".png";
			imwrite(ss.str(), currentCluster);
			++currentClusterIndex;
			currentImageIndex = 0;
		}
	};
}

#endif //HOG_FEATURES_AND_CLASSIFIER_MMP_DEBUG_IMAGE_UTILS