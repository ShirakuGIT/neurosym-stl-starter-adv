# PASTE THIS CODE INTO: src/ap_defs.py
import torch
from stl import always, eventually, until

class StlFormula:
    """
    A robust, general-purpose STL formula parser and evaluator.
    This replaces the fragmented, hardcoded formula classes.
    """
    def __init__(self, spec_dict, formula_str):
        self.spec_dict = spec_dict
        self.formula_str = formula_str
        self.ap_names = list(spec_dict.keys())
        self.ap_dict = {name: i for i, name in enumerate(self.ap_names)}
        self.env = None # This will be set by the Env class

    def get_cond_vec(self):
        """Generates a conditioning vector from the specification dictionary."""
        cond_vec = []
        # A consistent order is crucial for the model
        for ap_name in sorted(self.spec_dict.keys()):
            val = self.spec_dict[ap_name]
            if isinstance(val, list): # Handle list of obstacles
                for item in val:
                    if 'center' in item:
                        cond_vec.extend(item['center'].tolist())
            elif isinstance(val, dict) and 'center' in val:
                cond_vec.extend(val['center'].tolist())
        return torch.tensor(cond_vec, dtype=torch.float32)

    def get_robs(self, traj_batch):
        """
        Computes the robustness score for a batch of trajectories.
        This now uses the environment's specific ap_map function.
        """
        if self.env is None:
            raise ValueError("Environment not set for this specification.")
        
        # ap_values shape: (B, T, num_aps)
        ap_values = self.env.ap_map(traj_batch)
        
        # Create a safe local scope for the formula evaluation
        local_scope = {name: ap_values[:, :, i] for name, i in self.ap_dict.items()}
        safe_scope = {
            'always': always,
            'eventually': eventually,
            'until': until,
            **local_scope
        }

        try:
            # Evaluate the formula string within the safe scope
            robs = eval(self.formula_str, {"__builtins__": {}}, safe_scope)
            return robs
        except Exception as e:
            raise NotImplementedError(f"Could not parse formula '{self.formula_str}'. Error: {e}")