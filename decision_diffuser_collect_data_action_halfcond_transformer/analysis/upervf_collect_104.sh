#!/bin/bash 

# cuda=("5" "6")
task=("halfcheetah-medium-v2" "halfcheetah-medium-replay-v2" "halfcheetah-medium-expert-v2" "hopper-medium-v2" "hopper-medium-replay-v2" "hopper-medium-expert-v2" "walker2d-medium-v2" "walker2d-medium-replay-v2" "walker2d-medium-expert-v2")
collect_num=50000
diff_path=(/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-v2/horizon_20_cond_length_2/24-0606-225729/checkpoint # halfcheetah medium 2
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-v2/horizon_20_cond_length_5/24-0606-225727/checkpoint # halfcheetah medium 5
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-v2/horizon_20_cond_length_10/24-0606-225725/checkpoint # halfcheetah medium 10
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-replay-v2/horizon_20_cond_length_2/24-0608-210615/checkpoint # halfcheetah medium replay 2
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-replay-v2/horizon_20_cond_length_5/24-0608-210615/checkpoint # halfcheetah medium replay 5
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-replay-v2/horizon_20_cond_length_10/24-0608-210615/checkpoint # halfcheetah medium replay 10
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-expert-v2/horizon_20_cond_length_2/24-0608-210615/checkpoint # halfcheetah medium expert 2
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-expert-v2/horizon_20_cond_length_5/24-0608-210615/checkpoint # halfcheetah medium expert 5
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-expert-v2/horizon_20_cond_length_10/24-0608-210615/checkpoint # halfcheetah medium expert 10
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-v2/horizon_20_cond_length_2/24-0606-225735/checkpoint # hopper medium 2
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-v2/horizon_20_cond_length_5/24-0606-225733/checkpoint # hopper medium 5 
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-v2/horizon_20_cond_length_10/24-0606-225731/checkpoint # hopper medium 8
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-replay-v2/horizon_20_cond_length_2/24-0608-210615/checkpoint # hopper medium replay 2
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-replay-v2/horizon_20_cond_length_5/24-0608-210615/checkpoint # hopper medium replay 5
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-replay-v2/horizon_20_cond_length_10/24-0608-210615/checkpoint # hopper medium replay 10
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-expert-v2/horizon_20_cond_length_2/24-0608-210615/checkpoint # hopper medium expert 2
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-expert-v2/horizon_20_cond_length_5/24-0608-210615/checkpoint # hopper medium expert 6
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/hopper-medium-expert-v2/horizon_20_cond_length_10/24-0608-210615/checkpoint # hopper medium expert 10
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-v2/horizon_20_cond_length_2/24-0618-143025/checkpoint # walker2d medium 2
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-v2/horizon_20_cond_length_5/24-0618-143025/checkpoint # walker2d medium 5
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-v2/horizon_20_cond_length_10/24-0618-143025/checkpoint # walker2d medium 10
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-replay-v2/horizon_20_cond_length_2/24-0608-210615/checkpoint # walker2d medium replay 2
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-replay-v2/horizon_20_cond_length_5/24-0608-210615/checkpoint # walker2d medium replay 5
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-replay-v2/horizon_20_cond_length_10/24-0608-210615/checkpoint # walker2d medium replay 10
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-expert-v2/horizon_20_cond_length_2/24-0608-210615/checkpoint # walker2d medium expert 2
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-expert-v2/horizon_20_cond_length_5/24-0608-210615/checkpoint # walker2d medium expert 5
           /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/half_cond_diffusion/walker2d-medium-expert-v2/horizon_20_cond_length_10/24-0608-210615/checkpoint # walker2d medium expert 10
            )
upervf_path=(/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221347/uper_value_func_checkpoint.pt # halfcheetah medium 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221351/uper_value_func_checkpoint.pt # halfcheetah medium 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221355/uper_value_func_checkpoint.pt # halfcheetah medium 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-replay-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221400/uper_value_func_checkpoint.pt # halfcheetah medium replay 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-replay-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221403/uper_value_func_checkpoint.pt # halfcheetah medium replay 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-replay-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221408/uper_value_func_checkpoint.pt # halfcheetah medium replay 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-expert-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221414/uper_value_func_checkpoint.pt # halfcheetah medium expert 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-expert-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221422/uper_value_func_checkpoint.pt # halfcheetah medium expert 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/halfcheetah-medium-expert-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221425/uper_value_func_checkpoint.pt # halfcheetah medium expert 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221311/uper_value_func_checkpoint.pt # hopper medium 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221315/uper_value_func_checkpoint.pt # hopper medium 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221319/uper_value_func_checkpoint.pt # hopper medium 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-replay-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221323/uper_value_func_checkpoint.pt # hopper medium replay 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-replay-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221327/uper_value_func_checkpoint.pt # hopper medium replay 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-replay-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221332/uper_value_func_checkpoint.pt # hopper medium replay 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-expert-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221335/uper_value_func_checkpoint.pt # hopper medium expert 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-expert-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221339/uper_value_func_checkpoint.pt # hopper medium expert 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/hopper-medium-expert-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221343/uper_value_func_checkpoint.pt # hopper medium expert 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221427/uper_value_func_checkpoint.pt # walker2d medium 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221428/uper_value_func_checkpoint.pt # walker2d medium 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221433/uper_value_func_checkpoint.pt # walker2d medium 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-replay-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221436/uper_value_func_checkpoint.pt # walker2d medium replay 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-replay-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221440/uper_value_func_checkpoint.pt # walker2d medium replay 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-replay-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221445/uper_value_func_checkpoint.pt # walker2d medium replay 10
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-expert-v2/er_0.95cond_length2_layer_3_head_1/24-0618-221450/uper_value_func_checkpoint.pt # walker2d medium expert 2
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-expert-v2/er_0.95cond_length5_layer_3_head_1/24-0618-221455/uper_value_func_checkpoint.pt # walker2d medium expert 5
             /home/liuzhihong/diffusion_related/diffusion_dt/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward_fixtime/walker2d-medium-expert-v2/er_0.95cond_length10_layer_3_head_1/24-0618-221500/uper_value_func_checkpoint.pt # walker2d medium expert 10
             )
upervf_type="fixtime_upervf"
horizon=20
model_index_map=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26)
cuda_index_map=(1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5 6 6 6 6 7 7 7)
env_map=(0 1 2 3 4 5 6 7 8)
cond_length=(2 5 10)


eval_file_path="/home/liuzhihong/diffusion_related/diffusion_dt/code/decision_diffuser_collect_data_action_halfcond_transformer/analysis/eval_store_upervf.py"

$i
$j
$total_index
$model_index
$cuda_index

for ((i=0;i<9;i++))
do
    for ((j=0;j<3;j++))
    do
        ((total_index = i * 3 + j))
        ((model_index = ${model_index_map[$total_index]}))
        ((cuda_index = ${cuda_index_map[$total_index]}))
        echo $total_index
        logfile=/home/liuzhihong/diffusion_related/diffusion_dt/exp_result/log/collect_data/log/store_traj_$total_index.txt
        CUDA_VISIBLE_DEVICES=$cuda_index python -u $eval_file_path\
                                                --task ${task[$i]}\
                                                --upervf_type $upervf_type\
                                                --cond_length ${cond_length[$j]}\
                                                --collect_num $collect_num\
                                                --loadpath ${diff_path[$model_index]}\
                                                --uper_vf_path ${upervf_path[$model_index]}\
                                                --horizon $horizon > $logfile 2>&1 &
        sleep 2

    done
done