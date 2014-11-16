#ifndef UTILS_HOG_CLASSIFICATION
#define UTILS_HOG_CLASSIFICATION

#include <sstream>

namespace Utils
{
	using namespace std;
	using namespace cv;

	template<typename TString>
	int str2int(TString str)
	{
		stringstream ss(str);
		int res;
		ss >> res;
		return res;
	}

	template<typename T>
	string int2str(T value)
	{
		stringstream ss;
		ss << value;
		return ss.str();
	}

	template<typename TRect>
	int GetIntersectionRectSquare(const TRect &rect1, const TRect &rect2)
	{
		int minX = max(rect1.x, rect2.x);
		int minY = max(rect1.y, rect2.y);

		int maxX = min(rect1.x + rect1.width, rect2.x + rect2.width);
		int maxY = min(rect1.y + rect1.height, rect2.y + rect2.height);

		int width = max(maxX - minX, 0);
		int height = max(maxY - minY, 0);

		return width*height;
	}

	template<typename T>
	inline T sqr(T x)
	{
		return x*x;
	}
}

#endif // UTILS_HOG_CLASSIFICATION