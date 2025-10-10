from resolve.network_model_architectures import HCTargetAttnNP, ConditionalNeuralProcess, HCTargetAttnLNP

class ModelsManager():
    def __init__(self, config):
        d_x = config["d_theta"] + config["d_phi"]
        d_y = config["d_y"]
        
        representation_size = config.get("representation_size", 128)
        encoder_sizes = config.get("encoder_sizes", [])
        decoder_sizes = config.get("decoder_sizes", [])
        
        encoder_sizes = [d_x + d_y] + (encoder_sizes if encoder_sizes else []) + [representation_size]
        decoder_sizes = [representation_size + d_x] + (decoder_sizes if decoder_sizes else []) + [d_y*2]
        
        self._models = {}
        self._models["HCTargetAttnNP"]=HCTargetAttnNP(config["d_theta"], config["d_phi"], d_y, representation_size)
        self._models["HCTargetAttnLNP"]=HCTargetAttnLNP(config["d_theta"], config["d_phi"], d_y, representation_size)
        self._models["ConditionalNeuralProcess"]=ConditionalNeuralProcess(encoder_sizes, decoder_sizes)

    def get_network(self, model_name):
        return self._models[model_name]