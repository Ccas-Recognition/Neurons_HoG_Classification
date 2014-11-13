#ifndef HOG_FEATURES_AND_CLASSIFIER_MMP_CONSTS
#define HOG_FEATURES_AND_CLASSIFIER_MMP_CONSTS

namespace HOGFeatureClassifier
{
	//best 77.48 {3, 6}, 8, norm all, without sum of edges
	//Context {
	const int blockSizeX[] = { 3, 6 };
	const int blockSizeY[] = { 3, 6 };
	const int blockSizeCount = sizeof(blockSizeX) / sizeof(int);
	const int dirSegSize = 8; 

	const int RESIZE_IMAGE_SIZE = 24;

	const float MODEL_THRESHOLD = 0.0;//4.55   
	const int nonlinear_n = 2; //3
	const float nonlinear_L = 0.5;
	const double param_C = 0.08;
	//}
}

#endif