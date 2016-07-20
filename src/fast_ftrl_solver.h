// Copyright (c) 2014-2015 The AsyncFTRL Project
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_FAST_FTRL_SOLVER_H
#define SRC_FAST_FTRL_SOLVER_H

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <map>
#include <omp.h>
#include "src/ftrl_solver.h"
#include "src/lock.h"
#include <chrono>

typedef std::chrono::milliseconds TimeT;
extern const double rand_val ;
enum { kParamGroupSize = 10, kFetchStep = 6, kPushStep = 6 };

inline size_t calc_group_num(size_t n) {
	return (n + kParamGroupSize - 1) / kParamGroupSize;
}

template<typename T>
class FtrlParamServer : public FtrlSolver<T> {
public:
	FtrlParamServer();

	virtual ~FtrlParamServer();

	virtual bool Initialize(
		T alpha,
		T beta,
		T l1,
		T l2,
		size_t n,
		T dropout = 0);

	virtual bool Initialize(const char* path);

	bool FetchParamGroup(T* n, T* z,T** nv,T** zv, T** v,size_t group);

	bool FetchParam(T* n, T* z,T** nv,T** zv,T** v);

	bool PushParamGroup(T* n, T* z,T** nv,T** zv, size_t group);

	inline bool is_start_from_last_model(){
		return start_from_last_model_;
	};
	inline const char* get_last_model(){
		return last_model_.c_str();
	};

private:
	size_t param_group_num_;
	SpinLock* lock_slots_;

	std::string last_model_;
	bool start_from_last_model_;
};

template<typename T>
class FtrlWorker : public FtrlSolver<T> {
public:
	FtrlWorker();

	virtual ~FtrlWorker();

	bool Initialize(
		FtrlParamServer<T>* param_server,
		size_t push_step = kPushStep,
		size_t fetch_step = kFetchStep);

	bool Reset(FtrlParamServer<T>* param_server);

	bool Initialize(
		T alpha,
		T beta,
		T l1,
		T l2,
		size_t n,
		T dropout = 0) { return false; }

	bool Initialize(const char* path) { return false; }

	T Update(const std::vector<std::pair<size_t, T> >& x, T y) { return false; }

	T Update(
		const std::vector<std::pair<size_t, T> >& x,
		T y,
		FtrlParamServer<T>* param_server);
	T Update(const std::vector<std::pair<size_t, T> >& xp,
				const std::vector<std::pair<size_t, T> >& xn,
				FtrlParamServer<T>* param_server);
	void update_para(std::map<size_t,float>& grad_w,
				std::map<size_t,std::vector<float> >& grad_v,T grad,
				FtrlParamServer<T>* param_server);
	bool PushParam(FtrlParamServer<T>* param_server);

private:
	size_t param_group_num_;
	size_t* param_group_step_;
	size_t push_step_;
	size_t fetch_step_;

	T * n_update_;
	T * z_update_;
	T** nv_update_;
	T** zv_update_;
};



template<typename T>
FtrlParamServer<T>::FtrlParamServer()
: FtrlSolver<T>(), param_group_num_(0), lock_slots_(NULL),start_from_last_model_(false) {}

template<typename T>
FtrlParamServer<T>::~FtrlParamServer() {
	if (lock_slots_) {
		delete [] lock_slots_;
	}
}

template<typename T>
bool FtrlParamServer<T>::Initialize(
		T alpha,
		T beta,
		T l1,
		T l2,
		size_t n,
		T dropout) {
	if (!FtrlSolver<T>::Initialize(alpha, beta, l1, l2, n, dropout)) {
		return false;
	}

	param_group_num_ = calc_group_num(n);
	lock_slots_ = new SpinLock[param_group_num_];

	FtrlSolver<T>::init_ = true;
	return true;
}

template<typename T>
bool FtrlParamServer<T>::Initialize(const char* path) {
	if (!FtrlSolver<T>::Initialize(path)) {
		return false;
	}

	start_from_last_model_ = true;
	last_model_ = std::string(path);

	param_group_num_ = calc_group_num(FtrlSolver<T>::feat_num_);
	lock_slots_ = new SpinLock[param_group_num_];

	FtrlSolver<T>::init_ = true;
	return true;
}

template<typename T>
bool FtrlParamServer<T>::FetchParamGroup(T* n, T* z,T** nv,T** zv,T** v, size_t group) {
	if (!FtrlSolver<T>::init_) return false;

	size_t start = group * kParamGroupSize;
	size_t end = std::min((group + 1) * kParamGroupSize, FtrlSolver<T>::feat_num_);

	std::lock_guard<SpinLock> lock(lock_slots_[group]);
	for (size_t i = start; i < end; ++i) {
		n[i] = FtrlSolver<T>::n_[i];
		z[i] = FtrlSolver<T>::z_[i];
        //add fm code
		if (FtrlSolver<T>::v_[i] != NULL && v[i] != NULL){
			for  (int j = 0; j < FtrlSolver<T>::l_dim_; ++j) { 
				v[i][j] = FtrlSolver<T>::v_[i][j];
				nv[i][j] = FtrlSolver<T>::nv_[i][j];
			}
		}
	}

	return true;
}

template<typename T>
bool FtrlParamServer<T>::FetchParam(T* n, T* z,T** nv,T** zv,T** v) {
	if (!FtrlSolver<T>::init_) return false;

	for (size_t i = 0; i < param_group_num_; ++i) {
		FetchParamGroup(n, z, nv,zv,v,i);
	}
	return true;
}

template<typename T>
bool FtrlParamServer<T>::PushParamGroup(T* n, T* z, T** nv,T** zv,size_t group) {
	if (!FtrlSolver<T>::init_) return false;

	size_t start = group * kParamGroupSize;
	size_t end = std::min((group + 1) * kParamGroupSize, FtrlSolver<T>::feat_num_);

	std::lock_guard<SpinLock> lock(lock_slots_[group]);
	for (size_t i = start; i < end; ++i) {
		FtrlSolver<T>::n_[i] += n[i];
		FtrlSolver<T>::z_[i] += z[i];

		n[i] = 0; z[i] = 0;
        if (!nv || !zv)
            continue;
		if (FtrlSolver<T>::v_[i] == NULL && FtrlSolver<T>::nv_[i] == NULL){
			FtrlSolver<T>::v_[i] = new T [FtrlSolver<T>::l_dim_];
			FtrlSolver<T>::nv_[i] = new T [FtrlSolver<T>::l_dim_];
			memset(FtrlSolver<T>::v_[i],0,sizeof(T) * FtrlSolver<T>::l_dim_);
			memset(FtrlSolver<T>::nv_[i],0,sizeof(T) * FtrlSolver<T>::l_dim_);
		}

		if (zv[i] != NULL && nv[i] != NULL){
			for  (int j = 0; j < FtrlSolver<T>::l_dim_; ++j) { 
				FtrlSolver<T>::v_[i][j] += zv[i][j];
				FtrlSolver<T>::nv_[i][j] += nv[i][j];
				nv[i][j] = 0.;  zv[i][j] = 0.;
			}
		}
	}
	return true;
}


template<typename T>
FtrlWorker<T>::FtrlWorker()
: FtrlSolver<T>(), param_group_num_(0), param_group_step_(NULL),
push_step_(0), fetch_step_(0), n_update_(NULL), z_update_(NULL) {}

template<typename T>
FtrlWorker<T>::~FtrlWorker() {
	if (param_group_step_) {
		delete [] param_group_step_;
	}

	if (n_update_) {
		delete [] n_update_;
	}

	if (z_update_) {
		delete [] z_update_;
	}
	if (zv_update_) {
	    for (size_t i = 0; i < FtrlSolver<T>::feat_num_; ++i) 
            if (zv_update_[i])
                delete [] zv_update_[i];
        delete [] zv_update_;
    }
	if (nv_update_) {
	    for (size_t i = 0; i < FtrlSolver<T>::feat_num_; ++i) 
            if (nv_update_[i])
                delete [] nv_update_[i];
        delete [] nv_update_;
    }
}

template<typename T>
bool FtrlWorker<T>::Initialize(
		FtrlParamServer<T>* param_server,
		size_t push_step,
		size_t fetch_step) {
    //multi worker shared one param_server,passed by pointer FtrlParamServer<T>* param_server
	FtrlSolver<T>::alpha_ = param_server->alpha();
	FtrlSolver<T>::beta_ = param_server->beta();
	FtrlSolver<T>::l1_ = param_server->l1();
	FtrlSolver<T>::l2_ = param_server->l2();
	FtrlSolver<T>::feat_num_ = param_server->feat_num();
	FtrlSolver<T>::dropout_ = param_server->dropout();
	FtrlSolver<T>::l_dim_ = param_server->l_dim();
	if (param_server->is_start_from_last_model()){
		if (!FtrlSolver<T>::Initialize(param_server->get_last_model())) 
			return false;
	}
	else{
		if (!FtrlSolver<T>::Initialize(FtrlSolver<T>::alpha_, FtrlSolver<T>::beta_, FtrlSolver<T>::l1_, 
			FtrlSolver<T>::l2_, FtrlSolver<T>::feat_num_, FtrlSolver<T>::dropout_)) {
			return false;
		}
	}


	n_update_ = new T[FtrlSolver<T>::feat_num_];
	z_update_ = new T[FtrlSolver<T>::feat_num_];
	zv_update_ = new T*[FtrlSolver<T>::feat_num_];
	nv_update_ = new T* [FtrlSolver<T>::feat_num_];
    for (size_t i = 0; i < FtrlSolver<T>::feat_num_; ++i){
        zv_update_[i] = NULL;
        nv_update_[i] = NULL;
    }

	set_float_zero(n_update_, FtrlSolver<T>::feat_num_);
	set_float_zero(z_update_, FtrlSolver<T>::feat_num_);

    FtrlSolver<T>::load_fm_index(); //load index participating in fm interaction

    printf("ftrl fea num:%ld\n",FtrlSolver<T>::feat_num_);
    printf("%d dim \n",FtrlSolver<T>::l_dim_);
	
	param_server->FetchParam(FtrlSolver<T>::n_, FtrlSolver<T>::z_,FtrlSolver<T>::nv_,FtrlSolver<T>::zv_,FtrlSolver<T>::v_);

	param_group_num_ = calc_group_num(FtrlSolver<T>::feat_num_);
	param_group_step_ = new size_t[param_group_num_];
	for (size_t i = 0; i < param_group_num_; ++i) 
		param_group_step_[i] = 0;

	push_step_ = push_step;
	fetch_step_ = fetch_step;

	FtrlSolver<T>::init_ = true;
	return FtrlSolver<T>::init_;
}

template<typename T>
bool FtrlWorker<T>::Reset(FtrlParamServer<T>* param_server) {
	if (!FtrlSolver<T>::init_) return 0;

	param_server->FetchParam(FtrlSolver<T>::n_, FtrlSolver<T>::z_,FtrlSolver<T>::nv_, FtrlSolver<T>::zv_,FtrlSolver<T>::v_);

	for (size_t i = 0; i < param_group_num_; ++i) {
		param_group_step_[i] = 0;
	}
	return true;
}

template<typename T>  
void FtrlWorker<T>::update_para(std::map<size_t,float>& grad_w,
				std::map<size_t,std::vector<float> >& grad_v,T grad,
				FtrlParamServer<T>* param_server){
	std::vector<size_t>idx_vec;
	for(auto it = grad_w.begin(); it != grad_w.end(); it++)
		idx_vec.push_back(it->first);
	size_t k = 0;

	for(k = 0;k < idx_vec.size();k++)
	{
		size_t i = idx_vec[k];
		size_t g = i / kParamGroupSize;
		if (param_group_step_[g] % fetch_step_ == 0) {
			param_server->FetchParamGroup(
				FtrlSolver<T>::n_,
				FtrlSolver<T>::z_,
				FtrlSolver<T>::nv_,
				FtrlSolver<T>::zv_,
				FtrlSolver<T>::v_,
				g);
		}
		T grad_i = grad_w[i];
        T w_i = FtrlSolver<T>::GetWeight(i);
        grad_i *= grad;
        T sigma = (sqrt(FtrlSolver<T>::n_[i] + grad_i * grad_i)
            - sqrt(FtrlSolver<T>::n_[i])) / FtrlSolver<T>::alpha_;
        FtrlSolver<T>::z_[i] += grad_i - sigma * w_i;
        FtrlSolver<T>::n_[i] += grad_i * grad_i;
        z_update_[i] += grad_i - sigma * w_i;
        n_update_[i] += grad_i * grad_i;
	
	
		T l2 = 0.01;
		if (grad_v.count(i) == 0){
			fprintf(stderr,"idx %lld  not in grad_v map\n",i);
			continue;
		}
		if (FtrlSolver<T>::v_[i] == NULL){
			FtrlSolver<T>::GetWeight(i,0);
		}
		if (zv_update_[i] == NULL){
			zv_update_[i]= new T[FtrlSolver<T>::l_dim_];
			nv_update_[i]= new T[FtrlSolver<T>::l_dim_];
			memset(zv_update_[i],0,sizeof(T) * FtrlSolver<T>::l_dim_);
			memset(nv_update_[i],0,sizeof(T) * FtrlSolver<T>::l_dim_);
		}
		std::vector<float>& v_g = grad_v[i];
		int j = 0;
        for (; j < FtrlSolver<T>::l_dim_; j++) {
                T gij = 0.;
                T vij = FtrlSolver<T>::v_[i][j];
				gij = grad * v_g[j]; 
                T v_gj_2 = pow(gij,2);
				T update_ij = -FtrlSolver<T>::alpha_ / (FtrlSolver<T>::beta_ + sqrt(FtrlSolver<T>::nv_[i][j])) * ( gij +  l2 * vij );
				FtrlSolver<T>::v_[i][j] +=  update_ij;
                FtrlSolver<T>::nv_[i][j] += v_gj_2;

                zv_update_[i][j] += update_ij;
                nv_update_[i][j] += v_gj_2;
        }

		if (param_group_step_[g] % push_step_ == 0) {
			param_server->PushParamGroup(n_update_, z_update_,nv_update_,zv_update_,g);
		}
		param_group_step_[g] += 1;	
	}

}


//rank with bayesian rank loss,objective function is 1. / exp(-wT(xp-xn));
template<typename T>
T FtrlWorker<T>::Update(const std::vector<std::pair<size_t, T> >& xp,const std::vector<std::pair<size_t, T> >& xn,FtrlParamServer<T>* param_server) {
	if (!FtrlSolver<T>::init_) return 0;
    
	std::map<size_t,float> grad_w;
	std::map<size_t,std::vector<float> > grad_v;


    int pos = -1; 
    T fp_val = FtrlSolver<T>::calc_func_val_opt(xp,grad_w,grad_v,pos);
    pos = 1; 
    T fn_val = FtrlSolver<T>::calc_func_val_opt(xn,grad_w,grad_v,pos);

    T diff = fp_val - fn_val;
	
	T pred = sigmoid(diff);
	T grad = 1 - pred; // gradient of  rank loss 

	update_para(grad_w,grad_v,grad,param_server);

	return pred;
}

//regression with logistic loss
template<typename T>
T FtrlWorker<T>::Update(
		const std::vector<std::pair<size_t, T> >& x,
		T y,
		FtrlParamServer<T>* param_server) {
	if (!FtrlSolver<T>::init_) return 0;

	std::map<size_t,float> grad_w;
	std::map<size_t,std::vector<float> > grad_v;

    int pos = 1; 
    T f_val = FtrlSolver<T>::calc_func_val_opt(x,grad_w,grad_v,pos);

    T pred = sigmoid(f_val);
    T grad = pred - y;

	update_para(grad_w,grad_v,grad,param_server);

	return pred;
}

template<typename T>
bool FtrlWorker<T>::PushParam(FtrlParamServer<T>* param_server) {
	if (!FtrlSolver<T>::init_) return false;

	for (size_t i = 0; i < param_group_num_; ++i) {
		param_server->PushParamGroup(n_update_, z_update_, nv_update_,zv_update_,i);
	}

	return true;
}

#endif // SRC_FAST_FTRL_SOLVER_H
/* vim: set ts=4 sw=4 tw=0 noet :*/
