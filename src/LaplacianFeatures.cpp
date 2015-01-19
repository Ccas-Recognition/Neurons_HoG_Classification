#include "LaplacianFeatures.h"
#include<cmath>
#include<iomanip>

#define DEBUG_LAPLACIAN 0

using namespace std;
namespace ImageFeatures
{
	float laplacian_func(float x, float scale)
	{
		float sqr_scale = scale*scale;
		float sqr_x = x*x;
		float exp_value = expf(-0.5f*sqr_x / sqr_scale);
		return -exp_value / sqr_scale + sqr_x*exp_value / (sqr_scale*sqr_scale);
	}
		
	float ComputeScaleSpaceLaplacianFeature(int pos, const vector<float> &values, vector<float> &buffer, const HOGContext &context)
	{
		float max_laplacian = 0.0f;

		for (float _scale = context.lf_begin_scale; _scale <= context.lf_end_scale; _scale += context.lf_scale_step)
		{
			int scale = int(_scale);

			int mid_first = pos - scale;
			int mid_last = pos + scale;

			int left_first = max( mid_first - scale * 2 - 1, 0 );
			int left_last = max(mid_first - 1, 0);

			int right_first = min( mid_last + 1, int( values.size() - 1) );
			int right_last = min(mid_last + scale * 2 + 1, int(values.size() - 1));

			double left_sum = 0.0;
			int left_func_sum = 0;
			for (int i = left_first; i <= left_last; ++i)
			{
				left_sum += values[i];
				left_func_sum += 1;
			}
			if (left_func_sum != 0)
			left_sum /= left_func_sum;

			double mid_sum = 0.0;
			int mid_func_sum = 0;
			for (int i = mid_first; i <= mid_last; ++i)
			{
				mid_sum += values[i];
				mid_func_sum += 1;
			}
			if (mid_func_sum != 0)
				mid_sum /= mid_func_sum;

			double right_sum = 0.0;
			int right_func_sum = 0;
			for (int i = right_first; i <= right_last; ++i)
			{
				right_sum += values[i];
				right_func_sum += 1;
			}
			if (right_func_sum != 0)
				right_sum /= right_func_sum;

			float laplacian = right_sum + left_sum - 2 * mid_sum;
			buffer.push_back(laplacian);
			/*if (mid_sum < right_sum || mid_sum < left_sum)
				continue;

			if (abs(laplacian) > abs(max_laplacian))
				max_laplacian = laplacian;
			*/
			#if DEBUG_LAPLACIAN == 1
				cout << "Convolution " << ((scale * 2 + 1) * 3) << " " << pos <<endl;
				//cout << left_first << " " << left_last << " " << mid_first << " " << mid_last << " " << right_first << " " << right_last << endl;
				cout << (left_sum) << " " << (-2*mid_sum) << " " << (right_sum) << endl;
				cout << laplacian << endl;
			#endif
			/*
			for (int i = 0; i < values.size(); ++i)
			{
				float func_value = laplacian_func(float(pos - i), scale*values.size());
				sum += values[i] * func_value;
				#if 1
				cout << setw(12) << func_value << ": " << setw(12) << values[i] << endl;
				#endif
				func_sum += abs(func_value);
			}
			sum /= func_sum;
			*/
		}
		#if DEBUG_LAPLACIAN == 1
		cout << endl;
		#endif
		return max_laplacian;
	}
}
