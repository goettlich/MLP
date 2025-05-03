import torch
from models import RNN, MLP, NeuralODE
from integrators import IntegratorFactory

class ModelFactory:
    """
    Factory class to instantiate different models based on their names.
    """
    @staticmethod
    def get_model(
            model_name: str, d_in: int, d_out: int, 
            model_weights_fn: str = None, 
            train: bool = True, **kwargs) -> torch.nn.Module:
        
        if model_name == "NODE-MLP":
            integrator = IntegratorFactory().get_integrator(
                kwargs['integrator_name'], dt_solver=kwargs['dt_solver'])
            internal = MLP(d_in=d_in, d_out=d_out, hidden_layers=kwargs['hidden_layers'])
            if model_weights_fn is not None:
                internal.load_state_dict(torch.load(model_weights_fn))
            if train:
                internal.train()
            else:
                internal.eval()
            model = NeuralODE(func=internal, integrator=integrator)
        
        elif model_name == "RNN":
            model = RNN(d_in=d_in, d_out=d_out, d_hidden=kwargs['hidden_size'], n_layers=kwargs['num_layers'])
            if model_weights_fn is not None:
                model.load_state_dict(torch.load(model_weights_fn))
            if train:
                model.train()
            else:
                model.eval()
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        return model
    
    @staticmethod
    def get_model_statedict(model_name: str, model):
        if model_name == "NODE-MLP":
            return model.func.state_dict()
        elif model_name == "RNN":
            return model.state_dict()
    
    @staticmethod
    def get_model_input(model_name: str, x,t):
        inputs = {'NODE-MLP': (x[:,0], t[0]), 'RNN': (x,)}
        return inputs[model_name]
    
    @staticmethod
    def get_model_output(model_name: str, x):
        outputs = {'NODE-MLP': x.permute((1,0,2)), 'RNN': x}
        return outputs[model_name]