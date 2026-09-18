import argparse
import os

import torch

from model.roissd_penalized import RoiSSDPenalized
from tools.helpers.config_reader import load_config
from tools.helpers.pipeline import load_model, resolve_device


def _extract_model_state(checkpoint):
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description='Load RoiSSDPenalized from a checkpoint and resave the inner RoiSSD state dict.'
    )
    parser.add_argument(
        '--config',
        dest='config_path',
        default='config/voc.yaml',
        type=str,
        help='Path to the training config used to build the model.',
    )
    parser.add_argument(
        '--checkpoint',
        dest='checkpoint_path',
        default=None,
        type=str,
        help='Path to the checkpoint to read. Defaults to train_params.ckpt_name under trained_models/task_name.',
    )
    parser.add_argument(
        '--output',
        dest='output_path',
        default=None,
        type=str,
        help='Path to write the unwrapped state dict. Defaults to <checkpoint>.unwrapped.pth.',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Use strict state-dict loading when restoring the wrapper.',
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    train_config = config['train_params']

    model_task_path = os.path.join('trained_models', train_config['task_name'])
    checkpoint_path = args.checkpoint_path or os.path.join(model_task_path, train_config['ckpt_name'])
    output_path = args.output_path or f'{checkpoint_path}.unwrapped.pth'

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint does not exist: {checkpoint_path}')

    device = resolve_device('cpu')
    model = load_model(
        config=config,
        dataset=None,
        load_checkpoint=False,
        use_penalized_roissd=True,
        model_device=device,
    )
    if not isinstance(model, RoiSSDPenalized):
        raise TypeError(
            f'Expected RoiSSDPenalized from config {args.config_path!r}, got {type(model).__name__}'
        )

    print(f'Loading checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_model = _extract_model_state(checkpoint)
    model.load_state_dict(checkpoint_model, strict=args.strict)

    inner_model = model.model
    inner_state = inner_model.state_dict()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.save(inner_state, output_path)
    print(f'Saved unwrapped RoiSSD state dict to: {output_path}')
    print(f'Number of parameters saved: {len(inner_state)}')


if __name__ == '__main__':
    main()
