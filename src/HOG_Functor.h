#ifndef HOG_FUNCTOR_H
#define HOG_FUNCTOR_H

#include <vector>
#include <opencv2/core/core.hpp>

#ifndef _LIBLINEAR_H
struct feature_node
{
	int index;
	double value;
};
#endif

namespace HOGFeatureClassifier
{
	using namespace std;
	using namespace cv;
	
	struct TModel;

	struct HoGResponseFunctor
	{
		Mat image, modDir;
		auto_ptr< struct TModel > model;
	private:
		vector< float > features;
		vector< struct feature_node > classifier_features;
	public:
		bool InitModel(string model_name);
		bool Init(Mat& _image);
		float operator()(int pos_x, int pos_y, int width, int height);
	};
}
#endif // HOG_FUNCTOR_H