import argparse
import os
from model.roissd_mobilenet import RoiSSDMobileNet
from model.roissd import RoiSSD
from model.ssd import SSD
import torch
from torch.optim.lr_scheduler import MultiStepLR

from tools.helpers.config_reader import load_config


def build_model_from_config(config, dataset_config, train_config):
    model_name = str(train_config['model'])
    if model_name == 'ssd':
        return SSD(config=config['model_params'], num_classes=dataset_config['num_classes'])
    if model_name == 'roissd':
        return RoiSSD(config=config['model_params'], num_classes=dataset_config['num_classes'])
    if model_name == 'roissd-mobilenet':
        return RoiSSDMobileNet(config=config['model_params'], num_classes=dataset_config['num_classes'])
    raise ValueError(f'Unknown model name {model_name!r} in train_params.model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ssd training')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    args = parser.parse_args()
    
    # Read the config file #
    config = load_config(args.config_path)

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    model_task_path = os.path.join('trained_models', train_config['task_name'])
    model_checkpoint_path = os.path.join(model_task_path, train_config['ckpt_name'])

    # Load checkpoint if it exists (after creating optimizer and scheduler)
    if os.path.exists(model_checkpoint_path):
        print('Loading checkpoint as one exists')
        checkpoint = torch.load(model_checkpoint_path, map_location='cpu')

        model = build_model_from_config(config, dataset_config, train_config)
        optimizer = torch.optim.SGD(
            lr=train_config['lr'],
            params=model.parameters(),
            weight_decay=5E-4,
            momentum=0.9,
        )
        lr_scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.5)
        
        # Handle both old format (state_dict only) and new format (full checkpoint)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            print('Checkpoint contains full state (model, optimizer, scheduler). Resetting learning rate and epoch.')
            model.load_state_dict(checkpoint['model'])
        else:
            print('Checkpoint contains model state only (old format). Rewriting as full checkpoint with reset optimizer/scheduler.')
            model.load_state_dict(checkpoint)

        # Reset optimizer and scheduler states
        checkpoint_out = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'epoch': 0,
        }
        torch.save(checkpoint_out, model_checkpoint_path)
        torch.save(0, os.path.join(model_task_path, 'epoch.pth'))
        print(f"Learning rate reset to {train_config['lr']} and epoch reset to 0 in checkpoint '{model_checkpoint_path}'.")

    else:
        print(f"Checkpoint '{model_checkpoint_path}' does not exist. Cannot reset learning rate.")
        exit(1)