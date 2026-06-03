from dataset.imagenet_vid_raw import ImageNetVidRawDataset
from dataset.voc_raw import VOCRawDataset
from dataset.yolo_imagenet_vid import YoloImageNetVidDataset, YoloImageNetVidRawDataset
from tools.helpers.config_reader import load_config
from tools.helpers.pipeline import load_model
from tools.infer import infer_and_evaluate
from tools.train_samplers.roi_mixed_sampling import MixedBatchSampler, MixedCollateFn, RoiBatchProcessor
import torch
import argparse
import os
import numpy as np
import random
import csv
import torchvision
from tqdm import tqdm
from model.roissd import RoiSSD
from model.roissd_mobilenet import RoiSSDMobileNet
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from model.ssd import SSD

if not torch.cuda.is_available():
    raise Exception('CUDA not available')
else:
    print('Running on CUDA')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


class ImageNetNormalize:
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype, device=img.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype, device=img.device).view(3, 1, 1)
        return (img - mean) / std


def resolve_optional_weights_path(train_config, key_name):
    weights_path = str(train_config.get(key_name, '')).strip()
    if not weights_path:
        return None
    if os.path.exists(weights_path):
        return weights_path

    task_name = str(train_config.get('task_name', '')).strip()
    if task_name:
        candidate = os.path.join('trained_models', task_name, weights_path)
        if os.path.exists(candidate):
            return candidate
    return weights_path


def build_fcos_model(num_classes):
    return torchvision.models.detection.fcos_resnet50_fpn(
        weights=None,
        weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
        num_classes=num_classes,
    )


def prepare_targets_for_fcos(images, targets, device):
    prepared_targets = []
    for image, target in zip(images, targets):
        _, h, w = image.shape
        boxes = target['boxes'].float().clone()
        boxes[:, [0, 2]] *= float(w)
        boxes[:, [1, 3]] *= float(h)
        prepared_targets.append({
            'boxes': boxes.to(device, non_blocking=True),
            'labels': target['labels'].long().to(device, non_blocking=True),
        })
    return prepared_targets


def train(args):
    # Read the config file #
    config = load_config(args.config_path)
    print(config)
    #########################

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    if train_config.get('dataset', 'voc') == 'imagenet-vid':
        train_dataset = ImageNetVidRawDataset(
            split='train',
            train_data_root=dataset_config['train_data_root'],
            train_ann_root=dataset_config['train_ann_root'],
            test_data_root=dataset_config['test_data_root'],
            test_ann_root=dataset_config['test_ann_root'],
            im_size=dataset_config.get('im_size', 300),
            task=None,
        )
    elif str(train_config['dataset']) == 'yolo-imagenet-vid':
        train_dataset = YoloImageNetVidRawDataset(
                     'train',
                     yolo_dataset_yaml=dataset_config['yolo_dataset_yaml'],
                     im_size=dataset_config['im_size'])
    else:
        train_dataset = VOCRawDataset(
            split='train',
            im_sets=dataset_config['train_im_sets'],
            task=None,
        )

    batch_sampler = MixedBatchSampler(
        dataset=train_dataset,
        batch_size=train_config['batch_size'],
        stage=train_config.get('roi_mixed_stage', 2),
        drop_last=True,
        shuffle=True,
        seed=train_config['seed'],
    )

    model_name = str(train_config['model'])
    processor = RoiBatchProcessor(
        image_only_transform=None,
        normalize_transform=None if model_name == 'fcos' else ImageNetNormalize(),
    )

    collate_fn = MixedCollateFn(processor, return_mode=True)
    train_dataset_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=False
    )

    # Instantiate model and load checkpoint if present
    if model_name == 'fcos':
        model = build_fcos_model(num_classes=dataset_config['num_classes'])
    elif model_name in ('ssd', 'roissd', 'roissd-mobilenet'):
        model = load_model(
            config=config,
            dataset=None,
            load_checkpoint=False,
            use_penalized_roissd=False,
        )
    else:
        raise Exception('Unknown model name {}'.format(train_config['model']))
    
    model.to(device)

    if model_name == 'roissd-mobilenet' and hasattr(model, 'set_batch_norm_frozen'):
        model.set_batch_norm_frozen(
            freeze_backbone=train_config.get('freeze_backbone_bn', True),
            freeze_extra=train_config.get('freeze_extra_bn', False),
            train_affine=train_config.get('train_bn_affine', True),
        )

    # Check model weights for NaN at start of each epoch
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN/Inf found in parameter: {name}")
            print(f"  Shape: {param.shape}")
            print(f"  NaN count: {torch.isnan(param).sum().item()}")
            print(f"  Inf count: {torch.isinf(param).sum().item()}")
            raise RuntimeError("Model weights contain NaN/Inf - cannot continue training")
            
    model.train()
    
    model_task_path = os.path.join('trained_models', train_config['task_name'])
    model_checkpoint_path = os.path.join(model_task_path, train_config['ckpt_name'])
    if not os.path.exists(model_task_path):
        os.makedirs(model_task_path, exist_ok=True)

    optimizer = torch.optim.SGD(lr=train_config['lr'],
                                params=model.parameters(),
                                weight_decay=5E-4, momentum=0.9)
    
    # Get learning rate steps for the current stage of mixed sampling
    roi_stage = train_config.get('roi_mixed_stage', 1)
    _stage_suffix = f'_stage{roi_stage}' if roi_stage > 1 else ''
    _lr_steps_key = f'lr_steps{_stage_suffix}'
    _lr_steps = train_config.get(_lr_steps_key, train_config['lr_steps'])
    print(f"Using learning rate steps: {_lr_steps} for stage {roi_stage}")
    lr_scheduler = MultiStepLR(optimizer, milestones=_lr_steps, gamma=0.5)
    
    # Load checkpoint if it exists (after creating optimizer and scheduler)
    if os.path.exists(model_checkpoint_path):
        print('Loading checkpoint as one exists')
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        
        # Handle both old format (state_dict only) and new format (full checkpoint)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['scheduler'])
            print('Restored optimizer and scheduler state')
        else:
            # Old format - just model state_dict
            model.load_state_dict(checkpoint)
            print('Loaded model only (old checkpoint format)')

    else:
        print('No checkpoint found, starting training from scratch')
        custom_weights_path = None
        if model_name == 'fcos':
            custom_weights_path = resolve_optional_weights_path(train_config, 'fcos_weights')
        if custom_weights_path and os.path.exists(custom_weights_path):
            print(f'Loading custom initialization weights from {custom_weights_path}')
            checkpoint = torch.load(custom_weights_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)

    print(f"lr_scheduler.milestones: {lr_scheduler.milestones}")
    acc_steps = train_config['acc_steps']

    _epochs_key = f'num_epochs{_stage_suffix}'
    num_epochs = train_config.get(_epochs_key, train_config['num_epochs'])

    steps = 0

    i_start = 0
    epoch_path = os.path.join(model_task_path, 'epoch.pth')
    if os.path.exists(epoch_path):
        print('Loading checkpoint epoch as one exists')
        i_start = int(torch.load(epoch_path)) + 1
    print('Starting training from epoch {}'.format(i_start))

    import time
    for i in range(i_start, num_epochs):
        epoch_start_time = time.time()
        ssd_classification_losses = []
        ssd_localization_losses = []
        fcos_centerness_losses = []
        for idx, (images, targets, mode) in enumerate(tqdm(train_dataset_loader, desc='Training epoch {}'.format(i+1) )):

            if idx % 50 == 0:
                print(f"Batch mode: {mode}, image size: {tuple(images.shape)}")
                    # Asynchronous GPU transfer for faster throughput
            if model_name == 'fcos':
                image_list = [image.to(device, non_blocking=True) for image in images]
                targets = prepare_targets_for_fcos(images, targets, device)
                batch_losses = model(image_list, targets)
                loss = sum(batch_losses.values())
                classification_loss = batch_losses['classification']
                localization_loss = batch_losses['bbox_regression'] + batch_losses.get('bbox_ctrness', 0.0)
                if 'bbox_ctrness' in batch_losses:
                    fcos_centerness_losses.append(batch_losses['bbox_ctrness'].item())
            else:
                for target in targets:
                    try:
                        target['boxes'] = target['boxes'].float().to(device, non_blocking=True)
                        target['labels'] = target['labels'].long().to(device, non_blocking=True)
                    except Exception as e:
                        print(targets)
                        print(target)
                        raise e

                images = images.to(device, non_blocking=True)
                ignore_regions = None
                batch_losses, _ = model(images, targets, ignore_regions)
                classification_loss = batch_losses['classification']
                localization_loss = batch_losses['bbox_regression']
                loss = classification_loss + localization_loss

            # Check for NaN before combining losses
            if torch.isnan(classification_loss) or torch.isnan(localization_loss):
                print(f"NaN detected! Classification: {classification_loss.item()}, BBox: {localization_loss.item()}")
                print(f"Batch index: {idx}")
                print('Targets: {}'.format(targets))

            ssd_classification_losses.append(classification_loss.item())
            ssd_localization_losses.append(localization_loss.item())
            loss = loss / acc_steps
            loss.backward()

             # Check for NaN gradients before optimizer step
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"NaN/Inf gradient in: {name}")
                        has_nan_grad = True
            
            if has_nan_grad:
                print(f"Skipping optimizer step due to NaN gradients at batch {idx}")
                optimizer.zero_grad()
                continue

            if (idx + 1) % acc_steps == 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradient to prevent exploding gradient
                optimizer.step()
                optimizer.zero_grad()
            if steps % train_config['log_steps'] == 0:
                loss_output = ''
                loss_output += 'SSD Classification Loss : {:.4f}'.format(np.mean(ssd_classification_losses))
                loss_output += ' | SSD Localization Loss : {:.4f}'.format(np.mean(ssd_localization_losses))
                if model_name == 'fcos' and fcos_centerness_losses:
                    loss_output += ' | FCOS Centerness Loss : {:.4f}'.format(np.mean(fcos_centerness_losses))
                print(loss_output)
            if torch.isnan(loss):
                print('Loss is becoming nan. Exiting')
                exit(0)

            steps += 1
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()
        print('Learning rate for epoch {}: {:.6f}'.format(i+1, lr_scheduler.get_last_lr()[0]))
        epoch_time = time.time() - epoch_start_time
        epoch_minutes = epoch_time / 60
        print('Finished epoch {}/{}'.format(i+1, num_epochs))
        print('Epoch execution time: {:.2f} minutes'.format(epoch_minutes))
        # if isinstance(collate_fn, EpochAwareCollateFn):
        #     collate_fn.print_and_reset_stats(i)
        # else:
        #     print('No epoch-aware collate function, skipping stats reset. collate_fn type: {}'.format(type(collate_fn)))
        
        loss_output = ''
        loss_output += 'SSD Classification Loss : {:.4f}'.format(np.mean(ssd_classification_losses))
        loss_output += ' | SSD Localization Loss : {:.4f}'.format(np.mean(ssd_localization_losses))
        print(loss_output)
        
        # Save full checkpoint with optimizer and scheduler state
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'epoch': i
        }
        torch.save(checkpoint, model_checkpoint_path)
        torch.save(i, os.path.join(model_task_path, 'epoch.pth'))
        
        # Save losses to CSV file
        csv_file_path = os.path.join(model_task_path, 'training_losses.csv')
        file_exists = os.path.exists(csv_file_path)
        
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writerow(['epoch', 'classification_loss', 'detection_loss', 'learning_rate'])
            
            # Write current epoch data
            writer.writerow([i+1, np.mean(ssd_classification_losses), np.mean(ssd_localization_losses), lr_scheduler.get_last_lr()[0]])
    print('Done Training...')
    print('Evaluating...')
    args.infer_samples = True
    args.evaluate = True
    args.results_path = os.path.join(model_task_path, dataset_config['transform_name'] + '_results')
    infer_and_evaluate(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ssd training')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    args = parser.parse_args()
    train(args)
