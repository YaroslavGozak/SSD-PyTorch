import argparse
import os
import torch
from torch.optim.lr_scheduler import MultiStepLR

from tools.helpers.config_reader import load_config
from tools.helpers.pipeline import load_model


def reset_learning_rate(config_path, stage=None):
    # Read the config file #
    config = load_config(config_path)

    train_config = config['train_params']

    model_task_path = os.path.join('trained_models', train_config['task_name'])
    model_checkpoint_path = os.path.join(model_task_path, train_config['ckpt_name'])

    # Load checkpoint if it exists (after creating optimizer and scheduler)
    if os.path.exists(model_checkpoint_path):
        print(f'Loading checkpoint {model_checkpoint_path}')
        checkpoint = torch.load(model_checkpoint_path, map_location='cpu')

        model = load_model(config=config, dataset=None, load_checkpoint=False)
        model.to(device='cpu')
        lr = train_config['lr']
        lr_steps = train_config['lr_steps']
        optimizer = torch.optim.SGD(
            lr=lr,
            params=model.parameters(),
            weight_decay=5E-4,
            momentum=0.9,
        )
        lr_scheduler = MultiStepLR(optimizer, milestones=lr_steps, gamma=0.5)
        
        # Handle both old format (state_dict only) and new format (full checkpoint)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            print(f'Checkpoint contains full state (model, optimizer, scheduler). Resetting learning rate to {lr} with steps {lr_steps} and epoch to 0.')
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
        checkpoint_stage = stage
        if checkpoint_stage is None and isinstance(checkpoint, dict):
            checkpoint_stage = checkpoint.get('stage')
        if checkpoint_stage is not None:
            checkpoint_out['stage'] = int(checkpoint_stage)
        torch.save(checkpoint_out, model_checkpoint_path)
        torch.save(0, os.path.join(model_task_path, 'epoch.pth'))
        print(f"Learning rate reset to {train_config['lr']} and epoch reset to 0 in checkpoint '{model_checkpoint_path}'.")
        return True

    else:
        print(f"Checkpoint '{model_checkpoint_path}' does not exist. Cannot reset learning rate.")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ssd training')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    args = parser.parse_args()
    if not reset_learning_rate(args.config_path):
        exit(1)