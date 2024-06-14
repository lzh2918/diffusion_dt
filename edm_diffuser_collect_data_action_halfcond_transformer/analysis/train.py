if __name__ == '__main__':
    import os
    current_upup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import sys
    sys.path.append(current_upup_dir)
    # sys.path.append("..")
    
    from ml_logger import logger #, instr, needs_relaunch
    from analysis import RUN
    # import jaynes
    from scripts.train import main
    from config.locomotion_config import Config
    from params_proto.neo_hyper import Sweep
    import argparse
    import datetime

    class edm_cfg:
        num_steps_denoising: int = 10
        sigma_min: float = 2e-3
        # simga sampler
        sigma_max_training: float = 40
        # diffusion sampler
        sigma_max_sampler: float = 10
        rho: int = 7
        order: int = 1
        s_churn: float = 0
        s_tmin: float = 0
        s_tmax: float = float("inf")
        s_noise: float = 1
        scale: float = 1.478
        loc: float = -0.225
        sigma_offset_noise: float = 1.0
        sigma_data: float = 1.0

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="halfcheetah-medium-replay-v2")
    parser.add_argument("--group", type=str, default="test")
    parser.add_argument("--horizon",type=int, default=20)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--ar_inv", type=bool, default=False)
    parser.add_argument("--diffusion", type=str, default="models.EDMDiffusion")
    # model
    parser.add_argument("--model", type=str, default="models.DiT_EDM")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--transformer_deepth", type=int, default=12)
    parser.add_argument("--transformer_heads", type=int, default=6)
    parser.add_argument("--edm_cfg", type=edm_cfg, default=edm_cfg())
    # dataset
    parser.add_argument("--data_loader", type=str, default="datasets.SequenceHalfcondDataset")
    parser.add_argument("--cond_length", type=int, default=10)
    args = parser.parse_args()

    # save model dir 
    current_upupupup_dir = os.path.dirname(os.path.dirname(current_upup_dir))
    save_root_dir_path = os.path.join(current_upupupup_dir, "exp_result", "saved_model", "collect_data")
    timestamp = datetime.datetime.now().strftime("%y-%m%d-%H%M%S")
    save_model_path = os.path.join(save_root_dir_path, args.group, args.task, f"horizon_{args.horizon}_cond_length_{args.cond_length}",timestamp)
    parser.add_argument("--save_model_path", type=str, default=save_model_path)

    args = parser.parse_args()

    jsonl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"default_inv.jsonl")
    sweep = Sweep(RUN, Config).load(jsonl_path)

    # check dir 
    if not os.path.exists(args.save_model_path):
            os.makedirs(args.save_model_path)

    # for kwargs in sweep:
    #     logger.print(RUN.prefix, color='green')
    #     jaynes.config("local")
    #     thunk = instr(main, **kwargs)
    #     jaynes.run(thunk)

    # jaynes.listen()
    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        main(args, **kwargs)
