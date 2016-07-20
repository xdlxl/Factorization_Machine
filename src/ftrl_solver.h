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

#ifndef SRC_FTRL_SOLVER_H
#define SRC_FTRL_SOLVER_H

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <random>
#include <string>
#include <strstream>
#include <utility>
#include <vector>
#include <set>
#include <string.h>
#include <map>
#include <unordered_map>
#include <omp.h>
#include "src/util.h"

#define DEFAULT_ALPHA 0.01
#define DEFAULT_BETA 1.
#define DEFAULT_L1 1.
#define DEFAULT_L2 1.
const double rand_val = 0.1;
int dim = 8;

template<typename T>
class FtrlSolver {
public:
	FtrlSolver();

	virtual ~FtrlSolver();

	virtual bool Initialize(
		T alpha,
		T beta,
		T l1,
		T l2,
		size_t n,
		T dropout = 0,int k = 8);

	virtual bool Initialize(const char* path);

	virtual T Update(const std::vector<std::pair<size_t, T> >& x, T y);
	virtual T Update(const std::vector<std::pair<size_t, T> >& xp,const std::vector<std::pair<size_t, T> >& xn);
	virtual T Predict(const std::vector<std::pair<size_t, T> >& x);

	virtual bool SaveModelAll(const char* path);
	virtual bool SaveModel(const char* path);
	virtual bool SaveModelSparse(const char* path);
	virtual bool SaveModelDetail(const char* path);

public:
	T alpha() { return alpha_; }
	T beta() { return beta_; }
	T l1() { return l1_; }
	T l2() { return l2_; }
	int  l_dim() { return l_dim_; }
	size_t feat_num() { return feat_num_; }
	T dropout() { return dropout_; }
    bool check_fea_index(size_t i);

protected:
	enum {kPrecision = 8};

protected:
	T GetWeight(size_t idx);
	T GetWeight(size_t row,size_t col);
	T GetWeightSave(size_t row,size_t col);
    void load_fm_index();
    void set_float_rand(T** x, size_t n,const int l_dim, T val);
    void update_para(std::set<size_t>& idx_inter,std::vector<std::pair<size_t, T> >& weights_vec,std::vector<T>& gradients_vec,T* v_p,T* v_n,T* v,T grad,bool fisrt);
    T calc_func_val(std::set<size_t>& idx,const std::vector<std::pair<size_t, T> >& x,std::vector<std::pair<size_t, T> >& weights_vec,
            std::vector<T>& gradients_vec,T* v_g);
	T calc_func_val(const std::vector<std::pair<size_t, T> >& x,std::map<size_t,float>& grad_w,std::map<size_t,std::vector<float> >& grad_v,int postive_flag);
	T calc_func_val_opt(const std::vector<std::pair<size_t, T> >& x,std::map<size_t,float>& grad_w,std::map<size_t,std::vector<float> >& grad_v,int postive_flag);

protected:
	T alpha_;
	T beta_;
	T l1_;
	T l2_;
	size_t feat_num_;
	T dropout_;
    int l_dim_;

	T * n_;
	T * z_;
    //factorization machine 
    T ** v_;
    T ** nv_;
    T ** zv_;

	bool init_;

    std::unordered_map<size_t,bool> fm_index_map;
	std::mt19937 rand_generator_;
	std::uniform_real_distribution<T> uniform_dist_;
};



template<typename T>
FtrlSolver<T>::FtrlSolver()
: alpha_(0), beta_(0), l1_(0), l2_(0), feat_num_(0),
dropout_(0), n_(NULL), z_(NULL), v_(NULL),nv_(NULL),zv_(NULL),init_(false),
uniform_dist_(0.0, std::nextafter(1.0, std::numeric_limits<T>::max())) {}

template<typename T>
FtrlSolver<T>::~FtrlSolver() {
	if (n_) {
		delete [] n_;
	}

	if (z_) {
		delete [] z_;
	}
    if (v_ || nv_ || zv_){
	    for (size_t i = 0; i < feat_num_; ++i) 
        {
            if (v_[i] )
                delete [] v_[i];
            if (nv_[i] )
                delete [] nv_[i];
        }
		if (v_)
        	delete [] v_;
		if (nv_)
        	delete [] nv_;
    }
}

template<typename T>
void set_float_zero(T* x, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		x[i] = 0;
	}
}
template<typename T>
bool FtrlSolver<T>::check_fea_index(size_t i){
        if (fm_index_map.count(i) == 1)
            return fm_index_map[i];
}

template<typename T>
void FtrlSolver<T>::set_float_rand(T** x, size_t n,const int l_dim_, T val){
    if (val == 0.0)
     {
        for (size_t i = 0; i < n; ++i) {
			memset(x[i],0,sizeof(T) * l_dim_);
        }
         return;
     }
     std::random_device rd;
     std::mt19937 gen;
	 gen.seed(time(nullptr));
     std::default_random_engine generator;
     std::uniform_real_distribution<> distribution(-rand_val,rand_val);
	 for (size_t i = 0; i < n; ++i) {
	    for (size_t j = 0; j < l_dim_; ++j) {
            x[i][j] = distribution(generator);
	    }
    }
}

template<typename T>
bool FtrlSolver<T>::Initialize(
		T alpha,
		T beta,
		T l1,
		T l2,
		size_t n,
		T dropout,int k) {
	alpha_ = alpha;
	beta_ = beta;
	l1_ = l1;
	l2_ = l2;
	feat_num_ = n;
	dropout_ = dropout;
    l_dim_ = dim;

    load_fm_index();

	n_ = new T[feat_num_];
	z_ = new T[feat_num_];
	v_ = new T*[feat_num_];
	nv_ = new T*[feat_num_];
    for (size_t i = 0; i < feat_num_; ++i){
        v_[i] = NULL;
        nv_[i] = NULL; 
	}

	set_float_zero(n_, n);
	set_float_zero(z_, n);

	init_ = true;
	return init_;
}

template<typename T>
bool FtrlSolver<T>::Initialize(const char* path) {
	std::fstream fin;
	fin.open(path, std::ios::in);
	if (!fin.is_open()) {
		return false;
	}

	fin >> alpha_ >> beta_ >> l1_ >> l2_ >> feat_num_ >> dropout_ >> l_dim_;
	if (!fin || fin.eof()) {
		fin.close();
		return false;
	}

	z_ = new T[feat_num_];
	n_ = new T[feat_num_];

	v_ = new T*[feat_num_];
	nv_ = new T*[feat_num_];
    for (size_t i = 0; i < feat_num_; ++i) {
        v_[i] = NULL;
        nv_[i] = NULL;
	}
	for (size_t i = 0; i < feat_num_; ++i) {
		fin >> z_[i];
		fin >> n_[i];
		if (!fin || fin.eof()) {
			fin.close();
			return false;
		}
	}

	char flag = '0';
	size_t idx = -1;
	while(fin >> flag){
		fin >> idx;
		if (flag == 'v'){
			v_[idx] = new T[l_dim_]; 
			for (size_t j = 0; j < l_dim_; ++j) {
				fin >> v_[idx][j];
			}
		}
		else if (flag == 'n'){
			nv_[idx] = new T[l_dim_]; 
			for (size_t j = 0; j < l_dim_; ++j) {
				fin >> nv_[idx][j];
			}
		}
		else{
			fprintf(stderr,"unknown flag %c \n",flag);
		}
	}

    load_fm_index();
	fin.close();
	init_ = true;
	return init_;
}

template<typename T>
void FtrlSolver<T>::load_fm_index(){
        const char* fm_index_file = "./fm_index";

	    std::fstream fin;
        fin.open(fm_index_file, std::ios::in);
        for(size_t i = 0;i < feat_num_; i++)
            fm_index_map[i] = false;
        if (!fin.is_open()) {
            return;
        }

		//first line latend factor dimension	
		fin >> l_dim_;
        size_t st_indx,ed_indx;
        T indic;
        while(fin >> st_indx >> ed_indx >> indic)
        {
            if (indic == 1.)
                for(size_t i = st_indx;i < ed_indx; i++)
                    fm_index_map[i+1] = true;
            if (!fin || fin.eof()) {
                fin.close();
                break;
            }
        }
        return;
}

template<typename T>
T FtrlSolver<T>::GetWeight(size_t idx) {
	T sign = 1.;
	T val = 0.;
	if (z_[idx] < 0) {
		sign = -1.;
	}

	if (util_less_equal(sign * z_[idx], l1_)) {
		val = 0.;
	} else {
		val = (sign * l1_ - z_[idx]) / ((beta_ + sqrt(n_[idx])) / alpha_ + l2_);
	}

	return val;
}
template<typename T>
T FtrlSolver<T>::GetWeight(size_t row,size_t col) {
	if (v_[row] == NULL){
		v_[row] = new T[l_dim_];
		nv_[row] = new T[l_dim_];
    	for (size_t j = 0; j < l_dim_; ++j) {
			T init_val = rand_val * uniform_dist_(rand_generator_);
			v_[row][j] = init_val;
			nv_[row][j] = 0.;
		}
		if (GetWeight(row) == 0.)
			return 0.;
	}
	if (GetWeight(row) == 0.)
    		return 0.;
	else
		return v_[row][col];
}
template<typename T>
T FtrlSolver<T>::GetWeightSave(size_t row,size_t col) {
        return v_[row][col];
}

template<typename T>
void FtrlSolver<T>::update_para(std::set<size_t>& idx_inter,std::vector<std::pair<size_t, T> >& weights_vec,std::vector<T>& gradients_vec,T* v_p,T* v_n,T* v,T grad,bool first){
	return;
}
template<typename T>
T FtrlSolver<T>::calc_func_val(const std::vector<std::pair<size_t, T> >& x,std::map<size_t,float>& grad_w,std::map<size_t,std::vector<float> >& grad_v,int postive_flag){
    T f_val = 0.;

	std::vector<float> v_g(l_dim_,0.);
    T vTv_xTx = 0;
    for (size_t j = 0; j < l_dim_; ++j) {
            T vTx = 0;
            T v2x2 = 0;

            for (auto& item : x) {
                size_t i = item.first;
				if (i > feat_num_) break;

                T vij = v_[i][j];
                T tmp = item.second * vij;
                vTx += tmp;
                v2x2 += tmp * tmp;
             }
             v_g[j] = vTx;
             vTv_xTx +=vTx *vTx; 
             vTv_xTx -=v2x2; 

    }
    vTv_xTx *= 0.5;
    f_val += vTv_xTx;

	for (auto& item : x) {
		size_t idx = item.first;
		if (idx >= feat_num_) break;

		T x_i = item.second;
		T w_i = GetWeight(idx);
		grad_w[idx] += x_i * postive_flag;
		f_val += w_i * x_i;

		//calc grad of v
		if (grad_v.count(idx) == 0){
			grad_v[idx] = std::vector<float>(l_dim_,0.);
		}
    	for (size_t j = 0; j < l_dim_; ++j) {
			grad_v[idx][j] += x_i * (v_g[j] - x_i * v_[idx][j]) * postive_flag;
		}
	}
    return f_val;
}
template<typename T>
T FtrlSolver<T>::calc_func_val_opt(const std::vector<std::pair<size_t, T> >& x,std::map<size_t,float>& grad_w,std::map<size_t,std::vector<float> >& grad_v,int postive_flag){
    T f_val = 0.;
	std::vector<float> v_g(l_dim_,0.);

    T vTv_xTx = 0;
	T tmp1 = 0.; T tmp2 = 0.; T tmp3 = 0.;T tmp4 = 0.;
	size_t j = 0;
    for (j=0; j < l_dim_; j+=4) {
		T vTv_xTx1 =0,vTv_xTx2 =0,vTv_xTx3 =0 ,vTv_xTx4=0;
		T v2x2_1 =0,v2x2_2 =0,v2x2_3 =0 ,v2x2_4 =0;
	    for (auto& item : x) {
			if (item.first >= feat_num_) break;
            size_t i = item.first;
			T x_i = item.second ;

            tmp1 = x_i * GetWeight(i,j);   tmp2 = x_i * GetWeight(i,j+1);
			tmp3 = x_i * GetWeight(i,j+2);; tmp4 = x_i * GetWeight(i,j+3);

            vTv_xTx1 += tmp1; vTv_xTx2 += tmp2;
            vTv_xTx3 += tmp3; vTv_xTx4 += tmp4;

            v2x2_1 += tmp1 * tmp1; v2x2_2 += tmp2 * tmp2;
            v2x2_3 += tmp3 * tmp3; v2x2_4 += tmp4 * tmp4;
        }
		v_g[j]   = vTv_xTx1; v_g[j+1] = vTv_xTx2;
		v_g[j+2] = vTv_xTx3; v_g[j+3] = vTv_xTx4;

		vTv_xTx += pow(vTv_xTx1 , 2) + pow(vTv_xTx2 , 2) + pow(vTv_xTx3 , 2) + pow(vTv_xTx4 , 2); 
		vTv_xTx -= (v2x2_1 + v2x2_2 + v2x2_3 + v2x2_4); 
    }
	//for remainning dimension 
	for(;j < l_dim_;j++){
        T vTx = 0;
        T v2x2 = 0;
	    for (auto& item : x) {
			if (item.first >= feat_num_) break;
            size_t i = item.first;
            T tmp = item.second * v_[i][j];
            vTx += tmp;
            v2x2 += tmp * tmp;
         }
		 v_g[j]   = vTx;
         vTv_xTx += vTx *vTx; 
         vTv_xTx -= v2x2; 
	}

    vTv_xTx *= 0.5;
	f_val += vTv_xTx;

	for (auto& item : x) {
		size_t idx = item.first;
		if (idx >= feat_num_) break;

		T x_i = item.second;
		T w_i = GetWeight(idx);
		grad_w[idx] += x_i * postive_flag;
		f_val += w_i * x_i;

		//calc grad of v
		if (grad_v.count(idx) == 0){
			grad_v[idx] = std::vector<float>(l_dim_,0.);
		}
    	for (size_t j = 0; j < l_dim_; ++j) {
			grad_v[idx][j] += x_i * (v_g[j] - x_i * v_[idx][j]) * postive_flag;
		}
	}

    return f_val;
}

template<typename T>
T FtrlSolver<T>::Update(const std::vector<std::pair<size_t, T> >& xp,const std::vector<std::pair<size_t, T> >& xn) {
	return 0.;
}
template<typename T>
T FtrlSolver<T>::Update(const std::vector<std::pair<size_t, T> >& x, T y) {
	return 0.;
}

template<typename T>
T FtrlSolver<T>::Predict(const std::vector<std::pair<size_t, T> >& x) {
	if (!init_) return 0;

	T wTx = 0.;
	for (auto& item : x) {
		size_t idx = item.first;
		T val = GetWeight(idx);
		wTx += val * item.second;
	}
    //fm code
    T vTv_xTx = 0;
    for (size_t j=0; j < l_dim_; ++j) {
        T vTx = 0;
        T v2x2 = 0;
	    for (auto& item : x) {
            size_t i = item.first;
                T vij = GetWeight(i,j);
                vTx += vij;
                v2x2 += vij*vij; 
           }
           vTv_xTx +=vTx *vTx; 
           vTv_xTx -=v2x2; 
    }

    vTv_xTx *= 0.5;
    wTx += vTv_xTx; 
	T pred = sigmoid(wTx);
	return pred;
}

template<typename T>
bool FtrlSolver<T>::SaveModel(const char* path) {
	if (!init_) return false;

	std::fstream fout;
	std::ios_base::sync_with_stdio(false);
	fout.open(path, std::ios::out);

	if (!fout.is_open()) {
		return false;
	}

	fout << std::fixed << std::setprecision(kPrecision);

    //save feature dimension
    fout << feat_num_ << "\n";
    //save latent factor dimension
    fout << l_dim_ << "\n";
	for (size_t i = 0; i < feat_num_; ++i) {
		T w = GetWeight(i);
		fout << w << "\n";
	}
	for (size_t i = 0; i < feat_num_; ++i) {
		T w = GetWeight(i);
        T indx = 0.0;
        if (w > 0.0 || w < 0.0)
            indx = 1;
        for (size_t j = 0; j < l_dim_; ++j) {
            if (j != l_dim_-1)
                fout << v_[i][j]* indx << "\t";
            else
                fout << v_[i][j] * indx << "\n";
        }
	}

	fout.close();
	return true;
}

template<typename T>
bool FtrlSolver<T>::SaveModelSparse(const char* path) {
	if (!init_) return false;

	std::fstream fout;
	std::ios_base::sync_with_stdio(false);
	fout.open(path, std::ios::out);

	if (!fout.is_open()) {
		return false;
	}

	fout << std::fixed << std::setprecision(kPrecision);

    fout << feat_num_ << "\n";
    fout << l_dim_ << "\n";
	for (size_t i = 0; i < feat_num_; ++i) {
		T w = GetWeight(i);
		fout << w << "\n";
	}
	for (size_t i = 0; i < feat_num_; ++i) {
		T w = GetWeight(i);
        if (w > 0.0 || w < 0.0){
			fout << i << "\t";
			for (size_t j = 0; j < l_dim_; ++j) {
				if (j != l_dim_-1)
					fout << v_[i][j] << "\t";
				else
					fout << v_[i][j] << "\n";
			}
        }
	}

	fout.close();
	return true;
}

template<typename T>
bool FtrlSolver<T>::SaveModelDetail(const char* path) {
	if (!init_) return false;

	std::fstream fout;
	std::ios_base::sync_with_stdio(false);
	fout.open(path, std::ios::out);

	if (!fout.is_open()) {
		return false;
	}

	fout << std::fixed << std::setprecision(kPrecision);
	fout << alpha_ << "\t" << beta_ << "\t" << l1_ << "\t"
		<< l2_ << "\t" << feat_num_ << "\t" << dropout_ << "\t" << l_dim_ << "\n";

	for (size_t i = 0; i < feat_num_; ++i) {
		fout << z_[i] << "\n";
		fout << n_[i] << "\n";
	}

	for (size_t i = 0; i < feat_num_; ++i) {
		T w = GetWeight(i);
        if (w > 0.0 || w < 0.0){
			fout << 'v' << "\t" << i << "\t";
			for (size_t j = 0; j < l_dim_-1; ++j) {
				fout << v_[i][j] << "\t";
			}
			fout << v_[i][l_dim_-1] << "\n";
			fout << 'n' << "\t" <<  i << "\t";
			for (size_t j = 0; j < l_dim_-1; ++j) {
				fout << nv_[i][j] << "\t";
		    }
			fout << nv_[i][l_dim_-1] << "\n";
		}
    }

	fout.close();
	return true;
}

template<typename T>
bool FtrlSolver<T>::SaveModelAll(const char* path) {
	std::string model_detail = std::string(path) + ".save";
	return SaveModelSparse(path) && SaveModelDetail(model_detail.c_str());
}



template<typename T>
class FMModel {
public:
	FMModel();
	virtual ~FMModel();

	bool Initialize(const char* path);
	bool InitializeSparse(const char* path);

	T Predict(const std::vector<std::pair<size_t, T> >& x);
	T PredictLoop(const std::vector<std::pair<size_t, T> >& x);
	T* GetFmVector(size_t index) ;
	T GetWeight(size_t index) ;
	int GetLatentDim();
private:
	std::vector<T> model_;
	T**  v_;
    size_t feat_num_;
    int l_dim_;
	bool init_;
};

template<typename T>
FMModel<T>::FMModel() : v_(NULL),init_(false) {
    feat_num_ = 8100000;
    l_dim_ = 8;
}

template<typename T>
FMModel<T>::~FMModel() {

    if (v_){
	    for (size_t i = 0; i < feat_num_; ++i) 
            if (v_[i] )
                delete [] v_[i];
        delete [] v_;
    }
}

template<typename T>
bool FMModel<T>::Initialize(const char* path) {
	std::fstream fin;
	fin.open(path, std::ios::in);
	if (!fin.is_open()) {
		return false;
	}

    fin >> feat_num_ ; //get feature dimension
    fin >> l_dim_ ; //get latentfactor dimension
	T w;
    for (size_t i = 0; i < feat_num_; ++i) 
    {
        fin >> w ;
		model_.push_back(w);
	}
    v_ = new T*[feat_num_];
	for (size_t i = 0; i < feat_num_; ++i) {
        v_[i] = new T[l_dim_];
	    for (size_t j = 0; j < l_dim_; ++j) {
            fin >> v_[i][j];
        }
        if (!fin || fin.eof()) {
            fin.close();
            return false;
        }
	}

	fin.close();

	init_ = true;
	return init_;
}
template<typename T>
bool FMModel<T>::InitializeSparse(const char* path) {
	std::fstream fin;
	fin.open(path, std::ios::in);
	if (!fin.is_open()) {
		return false;
	}

    fin >> feat_num_ ; //get feature dimension
    fin >> l_dim_ ; //get latentfactor dimension
	T w;
	size_t indx = 0;
    for (size_t i = 0; i < feat_num_; ++i) {
        fin  >>  w ;
		model_.push_back(w);
	}
    v_ = new T*[feat_num_];
    for (size_t i = 0; i < feat_num_; ++i) 
		v_[i] = NULL;

	while(fin >> indx) {
        v_[indx] = new T[l_dim_];
	    for (size_t j = 0; j < l_dim_; ++j) {
            fin >> v_[indx][j];
        }
	}

	fin.close();
	init_ = true;

	return init_;
}

template<typename T>
T* FMModel<T>::GetFmVector(size_t index) {
	if (index < feat_num_)
		return v_[index];
	return NULL;
}
template<typename T>
T FMModel<T>::GetWeight(size_t index) {
	if (index < feat_num_)
		return model_[index];
	return 0.;
}
template<typename T>
int FMModel<T>::GetLatentDim() {
	return l_dim_;
}

template<typename T>
T FMModel<T>::Predict(const std::vector<std::pair<size_t, T> >& x) {
	if (!init_) {
		printf("model init failed !\n");
		return 0;
	}

	T wTx = 0.;
	for (auto& item : x) {
		if (item.first >= model_.size()) break;
		wTx += model_[item.first] * item.second;
	}

    T vTv_xTx = 0;
    for (size_t j=0; j < l_dim_; ++j) {
        T vTx = 0;
        T v2x2 = 0;
	    for (auto& item : x) {
            size_t i = item.first;
			if (i >= model_.size()) break;
			if (v_[i] == NULL) continue;
            T tmp = item.second * v_[i][j];
            vTx += tmp;
            v2x2 += tmp * tmp;
         }
         vTv_xTx +=vTx *vTx; 
         vTv_xTx -=v2x2; 
    }
    vTv_xTx *= 0.5;
    wTx += vTv_xTx; 
	return wTx;
}
template<typename T>
T FMModel<T>::PredictLoop(const std::vector<std::pair<size_t, T> >& x) {
	if (!init_) {
		printf("model init failed !\n");
		return 0;
	}

	T wTx = 0.;
	for (auto& item : x) {
		if (item.first >= model_.size()) break;
		wTx += model_[item.first] * item.second;
	}

    T vTv_xTx = 0;
	T tmp1 = 0.;
	T tmp2 = 0.; 
	T tmp3 = 0.;
	T tmp4 = 0.;
	size_t j = 0;
    for (j=0; j < l_dim_; j+=4) {
		T vTv_xTx1 =0,vTv_xTx2 =0,vTv_xTx3 =0 ,vTv_xTx4=0;
		T v2x2_1 =0,v2x2_2 =0,v2x2_3 =0 ,v2x2_4 =0;
	    for (auto& item : x) {
            size_t i = item.first;
			if (i >= model_.size()) break;
			if (v_[i] == NULL) continue;

			T x_i = item.second ;

            tmp1 = x_i * v_[i][j];
			tmp2 = x_i * v_[i][j+1];
			tmp3 = x_i * v_[i][j+2];
			tmp4 = x_i * v_[i][j+3];

            vTv_xTx1 += tmp1;
            vTv_xTx2 += tmp2;
            vTv_xTx3 += tmp3;
            vTv_xTx4 += tmp4;

            v2x2_1 += tmp1 * tmp1;
            v2x2_2 += tmp2 * tmp2;
            v2x2_3 += tmp3 * tmp3;
            v2x2_4 += tmp4 * tmp4;
         }
		 vTv_xTx += pow(vTv_xTx1 , 2) + pow(vTv_xTx2 , 2) + pow(vTv_xTx3 , 2) + pow(vTv_xTx4 , 2); 
		 vTv_xTx -= (v2x2_1 + v2x2_2 + v2x2_3 + v2x2_4); 
    }
	//for remainning dimension 
	for(;j < l_dim_;j++){
        T vTx = 0;
        T v2x2 = 0;
	    for (auto& item : x) {
			if (item.first >= model_.size()) break;
            size_t i = item.first;
            T tmp = item.second * v_[i][j];
            vTx += tmp;
            v2x2 += tmp * tmp;
         }
         vTv_xTx +=vTx *vTx; 
         vTv_xTx -=v2x2; 
	}

    vTv_xTx *= 0.5;
    wTx += vTv_xTx; 
	return wTx;
}
void split_trainfiles_order(const char* train_files_list,std::vector<std::string>& split_train_list,int num_threads){
		std::ifstream fin;
		fin.open(train_files_list);
		std::vector<std::string> train_files_vec;
		std::string line;
		while(getline(fin,line)){
			train_files_vec.push_back(line);
		}
		fin.close();

		if (train_files_vec.size() >= num_threads){
			int split_num = num_threads;
			int each_split_num = train_files_vec.size() / split_num;
			for(int i = 0; i < split_num; i++){
				std::strstream ss;	std::string istr;
				ss << i;	ss >> istr;
				int j;
				std::ofstream ofs;
				std::string ofiles = std::string(train_files_list) + "." + istr;
				ofs.open(ofiles.c_str());
				for(j = 0; j < train_files_vec.size(); j++){
					if ( j % split_num  == i){
							ofs << train_files_vec[j] << "\n";
					}
				}
				ofs.close();
				split_train_list.push_back(ofiles);
			}
		}
		else{
			int split_num = train_files_vec.size();
			for(int i = 0; i < split_num; i++){
				std::strstream ss;	std::string istr;
				ss << i;	ss >> istr;
				std::ofstream ofs;
				std::string ofiles = std::string(train_files_list) + "." + istr;
				ofs.open(ofiles.c_str());
				ofs << train_files_vec[i] << "\n";
				ofs.close();
				split_train_list.push_back(ofiles);
			}
			printf("file num less than threads, files num is %d\n",split_num);
		}
}

#endif // SRC_FTRL_SOLVER_H
/* vim: set ts=4 sw=4 tw=0 noet :*/
