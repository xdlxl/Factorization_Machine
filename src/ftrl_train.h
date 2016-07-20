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

#ifndef SRC_FTRL_TRAIN_H
#define SRC_FTRL_TRAIN_H

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include "src/fast_ftrl_solver.h"
#include "src/file_parser.h"
#include "src/ftrl_solver.h"
#include "src/stopwatch.h"

const int max_feat_num = 7000000;
const int DEFAULT_BATCH_SIZE = 50000;
template<typename T>
size_t read_problem_info(
	const char* train_file,
	bool read_cache,
	size_t& line_cnt,
	size_t num_threads = 0);

template<typename T, class Func>
T evaluate_file(const char* path, const Func& func_predict, size_t num_threads = 0);

template<typename T>
T calc_loss(T y, T pred) {
	T max_sigmoid = static_cast<T>(MAX_SIGMOID);
	T min_sigmoid = static_cast<T>(MIN_SIGMOID);
	T one = 1.;
	pred = std::max(std::min(pred, max_sigmoid), min_sigmoid);
	T loss = y > 0 ? -log(pred) : -log(one - pred);
	return loss;
}


template<typename T>
class FastFtrlTrainer {
public:
	FastFtrlTrainer();

	virtual ~FastFtrlTrainer();

	bool Initialize(
		size_t epoch,
		size_t num_threads = 0,
		size_t push_step = kPushStep,
		size_t fetch_step = kFetchStep,
        int is_rank = 0);

	bool Train(
		T alpha,
		T beta,
		T l1,
		T l2,
		T dropout,
		const char* model_file,
		const char* train_file);

	bool Train(
		const char* last_model,
		const char* model_file,
		const char* train_file);

protected:
	bool TrainImpl(
		const char* model_file,
		const char* train_file);

    bool LoadBatchSamples(FileParser<T>& file_parser,
          std::map<int,std::vector<std::vector<std::pair<size_t, T> > > >& train_samples,
          int batch_size,
          size_t& cnt);
    size_t get_feat_num();

private:
	size_t epoch_;
	size_t push_step_;
	size_t fetch_step_;
	size_t num_threads_;
	int is_rank_; // if set 0,using regression,set 1 using rank loss
	FtrlParamServer<T> param_server_;

	bool init_;
};
template<typename T>                                                                                                                                  
size_t FastFtrlTrainer<T>::get_feat_num() {                                                                                                           
    std::fstream fin;                                                                                                                                 
    size_t feat_num = 0;                                                                                                                              
    fin.open("./feat_num", std::ios::in);                                                                                                             
    if (!fin.is_open()) {                                                                                                                             
        printf("please supply a file name feat_num including (max feauture index)+2");                                                                
        return false;                                                                                                                                 
    }                                                                                                                                                 
    fin >> feat_num;                                                                                                                                  
    fin.close();                                                                                                                                      
    return feat_num;                                                                                                                                  
}      



template<typename T>
bool FastFtrlTrainer<T>::LoadBatchSamples(FileParser<T>& file_parser,
          std::map<int,std::vector<std::vector<std::pair<size_t, T> > > >& train_samples,
          int batch_size,size_t& cnt){
	cnt = 0;
	std::vector<std::pair<size_t, T> > x;
	int click = 0,impre = 0;
	while (file_parser.ReadSample(click,impre, x)) {
        for(int i = 0;i < click;i++){
            train_samples[1].push_back(x);
        }
        for(int i = 0;i < impre - click;i++){
			train_samples[0].push_back(x);
        }
        ++cnt;
  		if (cnt >= batch_size){
                break;
		}
   }
    if (cnt > 0 )
        return true;
    return false;
}


template<typename T>
FastFtrlTrainer<T>::FastFtrlTrainer()
: epoch_(0), push_step_(0),
fetch_step_(0), param_server_(), num_threads_(0), is_rank_(0),init_(false){ }

template<typename T>
FastFtrlTrainer<T>::~FastFtrlTrainer() {
}

template<typename T>
bool FastFtrlTrainer<T>::Initialize(
		size_t epoch,
		size_t num_threads,
		size_t push_step,
		size_t fetch_step,
		int is_rank) {
	epoch_ = epoch;
	push_step_ = push_step;
	fetch_step_ = fetch_step;
	is_rank_= is_rank;
	if (num_threads == 0) {
		num_threads_ = std::thread::hardware_concurrency();
	} else {
		num_threads_ = num_threads;
	}

	init_ = true;
	return init_;
}


template<typename T>
bool FastFtrlTrainer<T>::Train(
		T alpha,
		T beta,
		T l1,
		T l2,
		T dropout,
		const char* model_file,
		const char* train_file) {
	if (!init_) return false;

	size_t line_cnt = 0;
    size_t feat_num = get_feat_num();
	if (feat_num == 0) return false;

	if (!param_server_.Initialize(alpha, beta, l1, l2, feat_num, dropout)) {
		return false;
	}

	return TrainImpl(model_file, train_file);
}

template<typename T>
bool FastFtrlTrainer<T>::Train(
		const char* last_model,
		const char* model_file,
		const char* train_file) {
	if (!init_) return false;

	size_t line_cnt = 0;
    size_t feat_num = get_feat_num();
	if (feat_num == 0) return false;

	if (!param_server_.Initialize(last_model)) {
		return false;
	}

	return TrainImpl(model_file, train_file);
}

template<typename T>
bool FastFtrlTrainer<T>::TrainImpl(
		const char* model_file,
		const char* train_file) {
	if (!init_) return false;

	fprintf(
		stdout,
		"params={alpha:%.2f, beta:%.2f, l1:%.2f, l2:%.2f, dropout:%.2f, epoch:%zu}\n",
		static_cast<float>(param_server_.alpha()),
		static_cast<float>(param_server_.beta()),
		static_cast<float>(param_server_.l1()),
		static_cast<float>(param_server_.l2()),
		static_cast<float>(param_server_.dropout()),
		epoch_);
	std::vector<std::string> split_train_list;
	split_trainfiles_order(train_file,split_train_list,num_threads_);
	if(split_train_list.size() < num_threads_ )
		num_threads_ = split_train_list.size();


	FtrlWorker<T>* solvers = new FtrlWorker<T>[num_threads_];
	for (size_t i = 0; i < num_threads_; ++i) {
		solvers[i].Initialize(&param_server_, push_step_, fetch_step_);
	}

	auto predict_func = [&] (const std::vector<std::pair<size_t, T> >& x) {
		return param_server_.Predict(x);
	};

	StopWatch timer;
	for (size_t iter = 0; iter < epoch_; ++iter) {

		size_t count = 0;

		SpinLock lock;
		auto worker_func = [&] (size_t i) {


			std::vector<std::pair<size_t, T> > xp, xn;
			T y;
			FileParser<T> file_parser;
			file_parser.OpenFile(split_train_list[i].c_str());

			int	batch_size = DEFAULT_BATCH_SIZE;
			size_t local_count = 0;

			std::map<int,std::vector<std::vector<std::pair<size_t, T> > > > train_samples;
			train_samples[0] = std::vector<std::vector<std::pair<size_t, T> > >();
			train_samples[1] = std::vector<std::vector<std::pair<size_t, T> > >();

			while (LoadBatchSamples(file_parser,train_samples,batch_size,local_count)) {
				 T pred ;
				 size_t pos_size = train_samples[1].size();
				 size_t neg_size = train_samples[0].size();
				 std::random_device rd;
				 std::mt19937 gen(rd());
				 std::uniform_int_distribution<> pos_distri(0, pos_size-1);
				 std::uniform_int_distribution<> neg_distri(0, neg_size-1);
				 std::uniform_real_distribution<> real_distri(0,1);
				 int rank_pair_num =  batch_size;

				if (pos_size == 0){
					fprintf(stderr,"[ERROR] postive example num equals 0 \n");
					continue;
				}

				double pos_neg_ratio = (double)pos_size/(double)(pos_size + neg_size);
				int neg_pos_ratio = neg_size / pos_size;

				//rank loss
				if(is_rank_ == 1){
					int epoch =std::min<int>(5,int(1./pos_neg_ratio));
					for(int k = 0; k < epoch;k++){
						for (auto& xp : train_samples[1]){
							int n_idx = neg_distri(gen);
							xn = train_samples[0][n_idx];
							pred = solvers[i].Update(xp,xn,&param_server_);
						}
					}
				}
				else if (is_rank_== 0){
					int epoch =std::min<int>(10,int(1./pos_neg_ratio) );
						for (auto& xp : train_samples[1]){
							pred = solvers[i].Update(xp,1,&param_server_);
							for(int k = 0;k < neg_pos_ratio;k++){
								int n_idx = neg_distri(gen);
								xn = train_samples[0][n_idx];
								pred = solvers[i].Update(xn,0,&param_server_);
							}
						}
				}
				else{
				
					fprintf(stderr,"[ERROR],unkonwn flag value %f \n",is_rank_);	
				}


			local_count = batch_size;
            {
				std::lock_guard<SpinLock> lockguard(lock);
				count += local_count;
				if (count % DEFAULT_BATCH_SIZE == 0){
					fprintf(stdout,"epoch=%zu processed=[%d] \r",iter,count);
					fflush(stdout);
				}
			}
			train_samples[0].clear(); train_samples[1].clear();
		}
        solvers[i].PushParam(&param_server_);
		file_parser.CloseFile();

	};
		for (size_t i = 0; i < num_threads_; ++i) {
			solvers[i].Reset(&param_server_);
		}

		util_parallel_run(worker_func, num_threads_);

	}

	delete [] solvers;
	return param_server_.SaveModelAll(model_file);
}

#endif // SRC_FTRL_TRAIN_H
/* vim: set ts=4 sw=4 tw=0 noet :*/
