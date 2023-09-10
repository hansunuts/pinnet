import sys
import os
project_root = os.path.abspath('.')
if project_root not in sys.path:
    sys.path.append(project_root)

from dataclasses import dataclass
from dataclasses import fields
from dataclass_wizard import YAMLWizard

@dataclass
class BaseConfig(YAMLWizard):
    name:str
    
    def __init__(self, exp_name:str=None) -> None:
        super().__init__()
        if exp_name != None:
            self.load_from_exp(exp_name)
    
    def load_from_exp(self, exp_name:str, cfg_name:str=None):
        if cfg_name == None:
            cfg_name = f"{self.name}.yaml"
            
        cfg_path = os.path.join(f"{project_root}", "experiments", f"{exp_name}", "configs", cfg_name)
        return self.from_yaml_file(cfg_path)
        
    def generate_to_exp(self, exp_name:str, cfg_name:str=None):
        if cfg_name == None:
            cfg_name = f"{self.name}.yaml"
            
        path = os.path.join(f"{project_root}", f"experiments", f"{exp_name}", "configs")

        if not os.path.exists(path):
            os.makedirs(path)

        cfg_path = os.path.join(path, f"{cfg_name}")
        self.to_yaml_file(cfg_path)