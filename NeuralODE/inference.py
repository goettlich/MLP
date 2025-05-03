from omegaconf import OmegaConf
import os
from model_factory import ModelFactory

def inference(exp_dir, model_name, model_weights_fn):

    config = OmegaConf.load(os.path.join(exp_dir, 'config.yml'))
    dim = 2 if config.system == 'pendulum' else 3
    
    model = ModelFactory().get_model(
        model_name=model_name, 
        d_in=dim, 
        d_out=dim, 
        model_weights_fn=model_weights_fn, 
        train=False, 
        **config.model_cfg[model_name]
        )
    
    return model

exp_dir = "./eval/run_00"
model_name = 