#!/bin/bash 

cuda=("0" "1" "2" "3")
task=("halfcheetah-medium-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-medium-v2" "hopper-medium-replay-v2" "hopper-medium-expert-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
collect_num=50000
diff_path=(/home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-v2/horizon_20_cond_length_2/24-0606-225729/checkpoint # halfcheetah medium 2
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-v2/horizon_20_cond_length_5/24-0606-225727/checkpoint # halfcheetah medium 5
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-v2/horizon_20_cond_length_10/24-0606-225725/checkpoint # halfcheetah medium 10
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-replay-v2/horizon_20_cond_length_2/24-0608-210615/checkpoint # halfcheetah medium replay 2
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-replay-v2/horizon_20_cond_length_5/24-0608-210615/checkpoint # halfcheetah medium replay 5
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-replay-v2/horizon_20_cond_length_10/24-0608-210615/checkpoint # halfcheetah medium replay 10
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-expert-v2/horizon_20_cond_length_2/24-0608-210615/checkpoint # halfcheetah medium expert 2
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-expert-v2/horizon_20_cond_length_5/24-0608-210615/checkpoint # halfcheetah medium expert 5
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-expert-v2/horizon_20_cond_length_10/24-0608-210615/checkpoint # halfcheetah medium expert 10
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-v2/horizon_20_cond_length_2/24-0606-225735/checkpoint # hopper medium 2
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-v2/horizon_20_cond_length_5/24-0606-225733/checkpoint # hopper medium 5 
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-v2/horizon_20_cond_length_10/24-0606-225731/checkpoint # hopper medium 8
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-replay-v2/horizon_20_cond_length_2/24-0608-210615/checkpoint # hopper medium replay 2
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-replay-v2/horizon_20_cond_length_5/24-0608-210615/checkpoint # hopper medium replay 5
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-replay-v2/horizon_20_cond_length_10/24-0608-210615/checkpoint # hopper medium replay 10
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-expert-v2/horizon_20_cond_length_2/24-0608-210615/checkpoint # hopper medium expert 2
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-expert-v2/horizon_20_cond_length_5/24-0608-210615/checkpoint # hopper medium expert 6
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-expert-v2/horizon_20_cond_length_10/24-0608-210615/checkpoint # hopper medium expert 10
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-v2/horizon_20_cond_length_2/24-0618-143025/checkpoint # walker2d medium 2
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-v2/horizon_20_cond_length_5/24-0618-143025/checkpoint # walker2d medium 5
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-v2/horizon_20_cond_length_10/24-0618-143025/checkpoint # walker2d medium 10
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-replay-v2/horizon_20_cond_length_2/24-0608-210615/checkpoint # walker2d medium replay 2
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-replay-v2/horizon_20_cond_length_5/24-0608-210615/checkpoint # walker2d medium replay 5
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-replay-v2/horizon_20_cond_length_10/24-0608-210615/checkpoint # walker2d medium replay 10
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-expert-v2/horizon_20_cond_length_2/24-0608-210615/checkpoint # walker2d medium expert 2
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-expert-v2/horizon_20_cond_length_5/24-0608-210615/checkpoint # walker2d medium expert 5
           /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-expert-v2/horizon_20_cond_length_10/24-0608-210615/checkpoint # walker2d medium expert 10
            )
upervf_path=(/home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-v2/er_0.95cond_length2_layer_3_head_1/24-0613-133827/dt_checkpoint.pt # halfcheetah medium 2
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-v2/er_0.95cond_length5_layer_3_head_1/24-0613-133831/dt_checkpoint.pt # halfcheetah medium 5
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-v2/er_0.95cond_length10_layer_3_head_1/24-0613-133835/dt_checkpoint.pt # halfcheetah medium 10
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-replay-v2/er_0.95cond_length2_layer_3_head_1/24-0618-143728/uper_value_func_checkpoint.pt # halfcheetah medium replay 2
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-replay-v2/er_0.95cond_length5_layer_3_head_1/24-0618-143732/uper_value_func_checkpoint.pt # halfcheetah medium replay 5
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-replay-v2/er_0.95cond_length10_layer_3_head_1/24-0618-143736/uper_value_func_checkpoint.pt # halfcheetah medium replay 10
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-expert-v2/er_0.95cond_length2_layer_3_head_1/24-0618-151455/uper_value_func_checkpoint.pt # halfcheetah medium expert 2
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-expert-v2/er_0.95cond_length5_layer_3_head_1/24-0618-151459/uper_value_func_checkpoint.pt # halfcheetah medium expert 5
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-expert-v2/er_0.95cond_length10_layer_3_head_1/24-0618-152153/uper_value_func_checkpoint.pt # halfcheetah medium expert 10
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-v2/er_0.95_cond_length_2/24-0611-161744/uper_value_func_checkpoint.pt # hopper medium 2
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-v2/er_0.95_cond_length_5/24-0611-161752/uper_value_func_checkpoint.pt # hopper medium 5
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-v2/er_0.95_cond_length_10/24-0611-161800/uper_value_func_checkpoint.pt # hopper medium 10
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-replay-v2/er_0.95cond_length2_layer_3_head_1/24-0618-143703/uper_value_func_checkpoint.pt # hopper medium replay 2
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-replay-v2/er_0.95cond_length5_layer_3_head_1/24-0618-143707/uper_value_func_checkpoint.pt # hopper medium replay 5
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-replay-v2/er_0.95cond_length10_layer_3_head_1/24-0618-143711/uper_value_func_checkpoint.pt # hopper medium replay 10
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-expert-v2/er_0.95cond_length2_layer_3_head_1/24-0613-133751/dt_checkpoint.pt # hopper medium expert 2
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-expert-v2/er_0.95cond_length5_layer_3_head_1/24-0613-133755/dt_checkpoint.pt # hopper medium expert 5
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/hopper-medium-expert-v2/er_0.95cond_length10_layer_3_head_1/24-0613-133759/dt_checkpoint.pt # hopper medium expert 10
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/walker2d-medium-v2/er_0.95cond_length2_layer_3_head_1/24-0613-134337/dt_checkpoint.pt # walker2d medium 2
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/walker2d-medium-v2/er_0.95cond_length5_layer_3_head_1/24-0613-134341/dt_checkpoint.pt # walker2d medium 5
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/walker2d-medium-v2/er_0.95cond_length10_layer_3_head_1/24-0613-134345/dt_checkpoint.pt # walker2d medium 10
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/walker2d-medium-replay-v2/er_0.95cond_length2_layer_3_head_1/24-0618-143740/uper_value_func_checkpoint.pt # walker2d medium replay 2
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/walker2d-medium-replay-v2/er_0.95cond_length5_layer_3_head_1/24-0618-143744/uper_value_func_checkpoint.pt # walker2d medium replay 5
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/walker2d-medium-replay-v2/er_0.95cond_length10_layer_3_head_1/24-0618-143748/uper_value_func_checkpoint.pt # walker2d medium replay 10
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/walker2d-medium-expert-v2/er_0.95cond_length2_layer_3_head_1/24-0618-143752/uper_value_func_checkpoint.pt # walker2d medium expert 2
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/walker2d-medium-expert-v2/er_0.95cond_length5_layer_3_head_1/24-0618-143756/uper_value_func_checkpoint.pt # walker2d medium expert 5
             /home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/walker2d-medium-expert-v2/er_0.95cond_length10_layer_3_head_1/24-0618-143800/uper_value_func_checkpoint.pt # walker2d medium expert 10
             )
path_index=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26)
cuda_index=(0 0 0 0 0 0 1 1 1 1 1 1 1 2 2 2 2 2 2 2 3 3 3 3 3 3 3)
cond_length=(2 5 10)
horizon=20

eval_file_path="/home/liuzhihong/diffusion_dt_temp/code/diffusion_dt/decision_diffuser_collect_data_action_halfcond_transformer/analysis/eval_store_upervf.py"

$i
$j
$total_index
$model_index
$cuda_choose

for ((i=0;i<9;i++))
do
    for ((j=0;j<3;j++))
    do
        ((total_index = i * 3 + j))
        ((model_index = ${path_index[$total_index]}))
        ((cuda_choose = ${cuda_index[$total_index]}))
        logfile=/home/liuzhihong/diffusion_dt_temp/code/exp_result/log/store_traj_$total_index.txt
        echo $total_index
        CUDA_VISIBLE_DEVICES=${cuda[$cuda_choose]} python -u $eval_file_path\
                                                --task ${task[$i]}\
                                                --cond_length ${cond_length[$j]}\
                                                --collect_num $collect_num\
                                                --loadpath ${diff_path[$model_index]}\
                                                --uper_vf_path ${upervf_path[$model_index]}\
                                                --horizon $horizon > $logfile 2>&1 &
        sleep 2
    done
done