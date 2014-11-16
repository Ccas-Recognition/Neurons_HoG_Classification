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
		Mat integralImage;
		auto_ptr< struct TModel > model;
	private:
		vector< float > features;
		vector< struct feature_node > classifier_features;
		float ComputeFastPredict(int x, int y, int w, int h);
	public:
		int DEBUG_COUNT_OPERATOR;

		bool InitModel(string model_name);
		bool InitModel(const TModel *_model);
		bool Init(const Mat& _image, const Mat& _additionalImage);
		float operator()(int pos_x, int pos_y, int width, int height);
	};
}
#endif // HOG_FUNCTOR_H