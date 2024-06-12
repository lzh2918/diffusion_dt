if __name__ == '__main__':
    import sys
    sys.path.append("/data/user/liuzhihong/paper/big_model/diffusion/decision_diffuser") 
    from ml_logger import logger, instr, needs_relaunch
    from analysis import RUN
    import jaynes
    from scripts.evaluate_inv_parallel_test import evaluate
    from config.locomotion_config import Config
    from params_proto.neo_hyper import Sweep

    sweep = Sweep(RUN, Config).load("/data/user/liuzhihong/paper/big_model/diffusion/decision_diffuser/analysis/default_inv.jsonl")

    for kwargs in sweep:
        logger.print(RUN.prefix, color='green')
        evaluate(**kwargs)
        # jaynes.config("local")
        # thunk = instr(evaluate, **kwargs)
        # jaynes.run(thunk)

    # jaynes.listen()