# PASTE THIS CODE INTO: src/stl_eval.py
import torch
import numpy as np
from stl import always, eventually, until # CORRECTED IMPORT

class STLEvaluator:
    def __init__(self, spec_string, ap_dict):
        self.spec_string = spec_string
        self.ap_dict = ap_dict
    
    def evaluate(self, ap_vals):
        local_scope = {name: ap_vals[:, :, i] for name, i in self.ap_dict.items()}
        safe_scope = {
            'always': always,
            'eventually': eventually,
            'until': until,
            **local_scope
        }
        try:
            robs = eval(self.spec_string, {"__builtins__": {}}, safe_scope)
            return robs
        except Exception as e:
            raise NotImplementedError(f"Could not parse formula '{self.spec_string}'. Error: {e}")