train_data=train_files
test_data=test_files
echo 7000000 > feat_num
al=0.01
cnt=0
max=1
l2=1
rank=1
time ./ftrl_fm_train -f $train_data -m fm.model --thread 8 --alpha $al --l2 $l2 --l1 1 --is_rank $rank --epoch 1 
time ./ftrl_fm_predict -t $test_data -m fm.model -o auc_out
auc=`head auc_out`
echo $auc
