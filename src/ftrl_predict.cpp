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

#include <unistd.h>
#include <cstdlib>
#include <utility>
#include <vector>
#include "src/file_parser.h"
#include "src/ftrl_solver.h"
#include "src/util.h"

void print_usage(int argc, char* argv[]) {
	printf("Usage:\n");
	printf("\t%s -t test_file -m model -o output_file\n", argv[0]);
}
template<typename T>
bool LoadBatchSamples(FileParser<T>& file_parser,
          std::map<int,std::vector<std::vector<std::pair<size_t, T> > > >& train_samples,int batch_size)
{
	int cnt = 0;
	std::vector<std::pair<size_t, T> > x;
	int click = 0,impre = 0;
	while (file_parser.ReadSample(click,impre, x)) {
        for(int i = 0;i < click;i++)
        {
             train_samples[1].push_back(x);
        }
        for(int i = 0;i < impre - click;i++)
        {
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

double calc_auc(const std::vector<std::pair<double, unsigned> >& scores) {
	size_t num_pos = 0;
	size_t num_neg = 0;
	for (size_t i = 0; i < scores.size(); ++i) {
		if (scores[i].second == 1) {
			++num_pos;
		} else {
			++num_neg;
		}
	}

	if (num_pos == 0 || num_neg == 0) {
		return 0.;
	}

	size_t tp = 0;
	size_t fp = 0;
	double prev_tpr = 0.;
	double prev_fpr = 0.;

	double auc = 0.;
	for (size_t i = 0; i < scores.size(); ++i) {
		if (scores[i].second == 1) {
			++tp;
		} else {
			++fp;
		}

		if (static_cast<double>(fp) / num_neg != prev_fpr) {
			auc += prev_tpr * (static_cast<double>(fp) / num_neg - prev_fpr);
			prev_tpr = static_cast<double>(tp) / num_pos;
			prev_fpr = static_cast<double>(fp) / num_neg;
		}
	}

	return auc;
}

int main(int argc, char* argv[]) {
	int ch;

	std::string test_file;
	std::string model_file;
	std::string output_file;

	while ((ch = getopt(argc, argv, "t:m:o:h")) != -1) {
		switch (ch) {
		case 't':
			test_file = optarg;
			break;
		case 'm':
			model_file = optarg;
			break;
		case 'o':
			output_file = optarg;
			break;
		case 'h':
		default:
			print_usage(argc, argv);
			exit(0);
		}
	}

	if (test_file.size() == 0 || model_file.size() == 0 || output_file.size() == 0) {
		print_usage(argc, argv);
		exit(1);
	}

	FMModel<double> model;
	model.InitializeSparse(model_file.c_str());

	double y = 0.;
    int click,impre;
	std::vector<std::pair<size_t, double> > x;
	FILE* wfp = fopen(output_file.c_str(), "w");
	size_t cnt = 0, correct = 0;
	double loss = 0.;
	int num_threads = 16;

	std::vector<std::string> split_train_list;
	split_trainfiles_order(test_file.c_str(),split_train_list,num_threads);
	if(split_train_list.size() < num_threads )
		num_threads = split_train_list.size();

	std::vector<std::pair<double, unsigned> > pred_scores;

	int batch_size = 100000;
	int count = 0;
	SpinLock lock;

	auto worker_func = [&] (size_t i) {
		FileParser<double> file_parser;
		file_parser.OpenFile(split_train_list[i].c_str());

		size_t local_count = 0;
		std::map<int,std::vector<std::vector<std::pair<size_t, double> > > > train_samples;
		train_samples[0] = std::vector<std::vector<std::pair<size_t, double> > >();
		train_samples[1] = std::vector<std::vector<std::pair<size_t, double> > >();
		std::vector<std::pair<double, unsigned> > local_pred_scores;

		local_pred_scores.clear();
		while (LoadBatchSamples<double>(file_parser,train_samples,batch_size)){
			for( size_t i = 0; i < train_samples[0].size();i++){
				std::vector<std::pair<size_t, double> >& tx = train_samples[0][i];
				double pred = model.Predict(tx);
				{
					local_pred_scores.push_back(std::move(
					std::make_pair(pred, static_cast<unsigned>(0))));
				}
			}
			for( size_t i = 0; i < train_samples[1].size();i++){
				std::vector<std::pair<size_t, double> >& tx = train_samples[1][i];
				double pred = model.Predict(tx);
				{
					local_pred_scores.push_back(std::move(
					std::make_pair(pred, static_cast<unsigned>(1))));
				}
			}
			train_samples[0].clear();
			train_samples[1].clear();
			local_count = batch_size;
			{
					std::lock_guard<SpinLock> lockguard(lock);
					count += local_count;
			}
		}//while
			std::lock_guard<SpinLock> lockguard(lock);
			{
				for( size_t i = 0; i <local_pred_scores.size();i++){
					pred_scores.push_back(local_pred_scores[i]);
				}
			}
		file_parser.CloseFile(); 
	};

	util_parallel_run(worker_func, num_threads);


	std::sort(
		pred_scores.begin(),
		pred_scores.end(),
		[] (const std::pair<double, unsigned>& l, const std::pair<double, unsigned>& r) {
		    return l.first > r.first;
		}
	);
	double auc = calc_auc(pred_scores);

	fprintf(wfp,"%lf\n", auc);

	fclose(wfp);

	return 0;
}
/* vim: set ts=4 sw=4 tw=0 noet :*/
