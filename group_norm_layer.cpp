#include <algorithm>
#include <vector>

#include "caffe/layers/group_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	
template <typename Dtype>
void GroupNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	GroupNormParameter param = this->layer_param_.group_norm_param();			// get parameters from prototxt
	if (bottom[0]->num_axes() == 1){
		channels_ = 1;
	}
	else{
		eps_ = param.eps();
		group_ratio_ = param.group_ratio();
		channels_ = bottom[0]->shape(1);
		num_ = bottom[0]->shape(0);
		group_num_ = channels_ / group_ratio_;
	}
}

template <typename Dtype>
void GroupNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	if (bottom[0]->num_axes() >= 1)
		CHECK_EQ(bottom[0]->shape(1), channels_);
	top[0]->ReshapeLike(*bottom[0]);

	vector<int> sz;						
	sz.push_back(group_ratio_ * num_); 		// 共有 group_ratio_ * num_个mean_和varaince_
	mean_.Reshape(sz); 		
	variance_.Reshape(sz);

	int spatial_dim = bottom[0]->count() / (channels_*bottom[0]->shape(0));
	temp_.ReshapeLike(*bottom[0]);    // [N, C, H, W]
	x_norm_.ReshapeLike(*bottom[0]);  // [N, C, H, W]
	sz[0] = group_num_ * spatial_dim;				  
	group_sum_multiplier_.Reshape(sz);

	caffe_set(group_sum_multiplier_.count(), Dtype(1.), group_sum_multiplier_.mutable_cpu_data());
}


template <typename Dtype>
void GroupNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	// 1)先将bottom[0]的数据复制到top[0]中, 这里是复制到cpu_data()里
	const Dtype* bottom_data = bottom[0]->cpu_data();
	// 2)将top的数据导入到mutable_cpu_data中
	Dtype* top_data = top[0]->mutable_cpu_data();	
	//int num_= bottom[0]->shape(0);
	int spatial_dim = bottom[0]->count() / (bottom[0]->shape(0)*channels_);

	if (bottom[0] != top[0]) {			// copy bottom[0] to top[0]
		caffe_copy(bottom[0]->count(), bottom_data, top_data);
	}

	// compute mean 
	caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * group_ratio_, spatial_dim * group_num_,
		1. / (group_num_ * spatial_dim), bottom_data,
		group_sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());

	// subtract mean
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * group_ratio_, group_num_ * spatial_dim, 1, -1.,
		mean_.cpu_data(), group_sum_multiplier_.cpu_data(), 1., top_data);

	// compute variance using var(X) = E((X-EX)^2)
	caffe_mul(top[0]->count(), top[0]->cpu_data(), top[0]->cpu_data(), temp_.mutable_gpu_data());	// (X-EX)^2
	caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * group_ratio_, group_num_ * spatial_dim,   			// E((X-EX)^2)
		1. / (group_num_ * spatial_dim), temp_.cpu_data(), group_sum_multiplier_.cpu_data(), 0., variance_.mutable_cpu_data());

	//normalize variance
	caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
	caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5), variance_.mutable_cpu_data());

	//replicate variance to input size
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * group_ratio_, group_num_ * spatial_dim, 1, 1., variance_.cpu_data(),
		group_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());

	caffe_div(top[0]->count(), top_data, temp_.cpu_data(), top_data);

	caffe_copy(x_norm_.count(), top_data,
		x_norm_.mutable_cpu_data());
}

template <typename Dtype>
void GroupNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
	const Dtype* top_data = x_norm_.cpu_data();
	const Dtype* top_diff;
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

	if (bottom[0] != top[0]){
		top_diff = top[0]->cpu_diff();
	} else {
		caffe_copy(x_norm_.count(), top[0]->cpu_diff(), x_norm_.mutable_cpu_diff());
		top_diff = x_norm_.cpu_diff();
	}
	int spatial_dim = bottom[0]->count() / (bottom[0]->shape(0)*channels_);

	// if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
	//
	// dE(Y)/dX =
	//   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
	//     ./ sqrt(var(X) + eps)
	//
	// where \cdot and ./ are hadamard product and elementwise division,
	// respectively, dE/dY is the top diff, and mean/var/sum are all computed
	// along all dimensions except the channels dimension.  In the above
	// equation, the operations allow for expansion (i.e. broadcast) along all
	// dimensions except the channels dimension where required.
	
	// sum(dE / dY \cdot Y)
	caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
	caffe_cpu_gemv<Dtype>(CblasNoTrans, num_ * group_ratio_, group_num_ * spatial_dim, 1., 
		bottom_diff, group_sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());

	// reshape (broadcast) the above
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * group_ratio_, group_num_ * spatial_dim, 1, 1.,
		mean_.cpu_data(), group_sum_multiplier_.cpu_data(), 0., bottom_diff);

	// sum(dE/dY \cdot Y) \cdot Y
	caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

	// sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
	caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * group_ratio_, group_num_ * spatial_dim, 1., 
		top_diff, group_sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());

	// reshape (broadcast) the above to make 
	// sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * group_ratio_, group_num_ * spatial_dim, 1, 1.,
		mean_.cpu_data(), group_sum_multiplier_.cpu_data(), 1., bottom_diff);

	// dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
	caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff, 
		Dtype(-1. / (group_num_ * spatial_dim)), bottom_diff);

	// note: temp_ still contains sqrt(var(X) + eps), computed during the forward pass
	caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);	
}

#ifdef CPU_ONLY
STUB_GPU(GroupNormLayer);
#endif

INSTANTIATE_CLASS(GroupNormLayer);
REGISTER_LAYER_CLASS(GroupNorm);
}