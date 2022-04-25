from torch import load

class DLsolver:
    def __init__(self, pretrain_path):
        self.solver = load(open(pretrain_path, 'rb'))
    
    def decision(self, inp):
        return self.solver(inp).detach().numpy()
