import os
import yaml

class YTConfig:
    def __init__(self, config_path='config/ytbb.yaml'):
        self.config_path = config_path
        self.config = self.read_config()
        self.dataset_config = self.config['dataset_params']
        self.root_dir = self.dataset_config['root_dir']
        self.annotations_dir = os.path.join(self.root_dir, "SequenceAnnotations")
        self.ims_dir = os.path.join(self.root_dir, "ResizedSequences")
        self.dataset_params = self.config['dataset_params']
        self.model_params = self.config['model_params']

    def read_config(self):
        # Read the config file #
        with open(self.config_path, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
        return config