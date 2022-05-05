from torch import load
from algorithms.pred_model import GameGAT
from algorithms.DL_model.config.default import get_default_cfg
class DLsolver:
    def __init__(self, pretrain_path, cfg_path):
        cfg = get_default_cfg()
        cfg.merge_from_file(cfg_path)
        cfg.freeze()
        self.solver = GameGAT(cfg.MODEL.GAT)
        self.solver.load_state_dict(load(open(pretrain_path, 'rb')))
        self.solver.eval()

    def decision(self, inp):
        return self.solver(inp).detach().numpy()
