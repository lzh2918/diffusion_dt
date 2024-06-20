if __name__ == '__main__':
    import os
    current_upup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import sys
    sys.path.append(current_upup_dir)
    # sys.path.append("..")
    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    # import jaynes
    from scripts.evaluate_inv_parallel_store_upervf import evaluate
    from config.locomotion_config import Config
    from params_proto.neo_hyper import Sweep
    import argparse
    import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="halfcheetah-medium-replay-v2")
    # parser.add_argument("--group", type=str, default="test")
    # parser.add_argument("--data_loader", type=str, default="datasets.SequenceRewardDataset")
    parser.add_argument("--collect_num", type=int, default=1000)
    parser.add_argument("--data_loader", type=str, default="datasets.SequenceHalfcondTimestepDataset")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--loadpath", type=str, default="/home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/half_cond_diffusion/halfcheetah-medium-replay-v2/horizon_20_cond_length_5/24-0608-210615/checkpoint")
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--return_scale_high", type=float, default=1.2)
    parser.add_argument("--return_scale_low", type=float, default=1.0)
    parser.add_argument("--diffusion", type=str, default="models.GaussianDiffusion")
    # model
    parser.add_argument("--model", type=str, default="models.DiT")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--transformer_deepth", type=int, default=12)
    parser.add_argument("--transformer_heads", type=int, default=6)
    # half cond
    parser.add_argument("--cond_length", type=int, default=5)
    # uper vf
    parser.add_argument("--value_func_emb_dim", type=int, default=128)
    parser.add_argument("--max_episode_len", type=int, default=1000)
    parser.add_argument("--uper_vf_path", type=str, default="/home/liuzhihong/diffusion_dt_temp/code/exp_result/saved_model/collect_data/uper_value_func/halfcond_transformer_noreward/halfcheetah-medium-replay-v2/er_0.95cond_length5_layer_3_head_1/24-0618-143732/uper_value_func_checkpoint.pt")


    args = parser.parse_args()

    # save model dir 
    current_upupupup_dir = os.path.dirname(os.path.dirname(current_upup_dir))
    save_root_dir_path = os.path.join(current_upupupup_dir, "exp_result", "saved_model", "collect_data")
    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    load_diff_path_list = args.loadpath.split("/")
    save_traj_root_path = "/".join(load_diff_path_list[:-4])
    uper_vf_tag = args.uper_vf_path.split("/")[-3]
    uper_vf_date = args.uper_vf_path.split("/")[-2]
    save_traj_path = os.path.join(save_traj_root_path, "store_data", load_diff_path_list[-4], "diff_"+load_diff_path_list[-3],f"uper_vf_{uper_vf_tag}",f"diff_date_{load_diff_path_list[-2]}_upervf_date_{uper_vf_date}",timestamp)
    parser.add_argument("--save_traj_path", type=str, default=save_traj_path)

    args = parser.parse_args()

    jsonl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"default_inv.jsonl")
    sweep = Sweep(RUN, Config).load(jsonl_path)

    # check dir 
    if not os.path.exists(args.save_traj_path):
            os.makedirs(args.save_traj_path)

    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        evaluate(args, **kwargs)
        # jaynes.config("local")
        # thunk = instr(evaluate, **kwargs)
        # jaynes.run(thunk)

    # jaynes.listen()