from tools.helpers.config_reader import load_config
from tools.helpers.pipeline import load_dataset, load_model, resolve_device
from tools.infer import infer_and_evaluate
from tools.multiscale_collate import EpochAwareCollateFn
import torch
import argparse
import os
import numpy as np
import random
import csv
import torchvision
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

device = resolve_device(None)
print('Using device {}'.format(device))


def collate_function(data):
    return tuple(zip(*data))


def append_epoch_metrics_csv(
    csv_file_path,
    *,
    epoch,
    classification_loss,
    detection_loss,
    learning_rate,
    mean_ap,
    mean_detector_recall,
):
    file_exists = os.path.exists(csv_file_path)
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow([
                'epoch',
                'classification_loss',
                'detection_loss',
                'learning_rate',
                'mAP',
                'mean_detector_recall',
            ])
        writer.writerow([
            int(epoch),
            float(classification_loss),
            float(detection_loss),
            float(learning_rate),
            float(mean_ap),
            float(mean_detector_recall),
        ])


def train(args):
    # Read the config file #
    config = load_config(args.config_path)

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    print(f'train config', train_config)
    dataset = load_dataset(config, split='train')
    
    _fill = tuple(a + b for a, b in zip([123.0, 117.0, 104.0], (20, 20, 15)))  # correct colour
    if dataset_config['transform_name'] == 'no_resize_transform':
        collate_fn = EpochAwareCollateFn(
            num_epochs=train_config['num_epochs'],
            fill=_fill,
        )
    else:
        collate_fn = collate_function

    train_dataset_loader = DataLoader(dataset,
                               batch_size=train_config['batch_size'],
                               shuffle=True,
                               collate_fn=collate_fn,
                               num_workers=4,  # 0 - 1 process, 4 or 8 - number of processes
                               pin_memory=True,  # Add this for faster GPU transfer
                               persistent_workers=True, # Keep workers alive between epochs
                               prefetch_factor=2  # Prefetch 2 batches per worker
                               ) 

    model = load_model(
        config=config,
        dataset=None,
        load_checkpoint=False,
        use_penalized_roissd=False,
        model_device=device,
    )
    
    pretrained_detector = torchvision.models.detection.ssd300_vgg16(weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT)
    pretrained_detector.to(device)
    pretrained_detector.eval()
    
    model.to(device)
    model.train()

    if str(train_config['model']) == 'roissd-mobilenet' and hasattr(model, 'set_batch_norm_frozen'):
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
            
    model_task_path = os.path.join('trained_models', train_config['task_name'])
    model_checkpoint_path = os.path.join(model_task_path, train_config['ckpt_name'])
    if not os.path.exists(model_task_path):
        os.makedirs(model_task_path, exist_ok=True)

    optimizer = torch.optim.SGD(lr=train_config['lr'],
                                params=model.parameters(),
                                weight_decay=5E-4, momentum=0.9)
    lr_scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.5)
    
    # Load checkpoint if it exists (after creating optimizer and scheduler)
    if os.path.exists(model_checkpoint_path):
        print('Loading checkpoint as one exists')
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        
        # Handle both old format (state_dict only) and new format (full checkpoint)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            print('Restored optimizer and scheduler state')
        else:
            # Old format - just model state_dict
            model.load_state_dict(checkpoint)
            print('Loaded model only (old checkpoint format)')
        
        # Re-apply BN freeze after checkpoint loading to ensure frozen state persists
        if str(train_config['model']) == 'roissd-mobilenet' and hasattr(model, 'set_batch_norm_frozen'):
            model.set_batch_norm_frozen(
                freeze_backbone=train_config.get('freeze_backbone_bn', True),
                freeze_extra=train_config.get('freeze_extra_bn', False),
                train_affine=train_config.get('train_bn_affine', True),
            )

    else:
        print('No checkpoint found, starting training from scratch')
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    steps = 0

    i_start = 0
    epoch_path = os.path.join(model_task_path, 'epoch.pth')
    if os.path.exists(epoch_path):
        print('Loading checkpoint epoch as one exists')
        i_start = int(torch.load(epoch_path)) + 1
    print('Starting training from epoch {}'.format(i_start))

    import time
    for i in range(i_start, num_epochs):
        if isinstance(collate_fn, EpochAwareCollateFn):
            collate_fn.epoch = i  # Update epoch in collate function for dynamic multi-scale resizing
        epoch_start_time = time.time()
        ssd_classification_losses = []
        ssd_localization_losses = []
        for idx, (ims, targets, _) in enumerate(tqdm(train_dataset_loader, desc='Training epoch {}'.format(i+1) )):

            # Asynchronous GPU transfer for faster throughput
            for target in targets:
                try:
                    target['boxes'] = target['bboxes'].float().to(device, non_blocking=True)
                    del target['bboxes']
                    target['labels'] = target['labels'].long().to(device, non_blocking=True)
                except Exception as e:
                    print(targets)
                    print(target)
                    raise e
                
            # Stack images and transfer to GPU asynchronously
            images = torch.stack([im.float() for im in ims], dim=0).to(device, non_blocking=True)

            if dataset.__class__.__name__ == 'YTBBDataset':
                from model.roissd import generate_ignore_regions
                # 1. Run pre-trained detector (e.g., yolov5, coco-ssd) on images
                with torch.no_grad():
                    detector_outputs = []
                    detector_outputs_raw = pretrained_detector(images)  # returns list of dicts with 'boxes'
                    for i, det in enumerate(detector_outputs_raw):
                        boxes = det['boxes']
                        scores = det['scores']
                        keep = scores > 0.5
                        boxes = boxes[keep]
                        # Get image size for normalization
                        im = ims[i]
                        h, w = im.shape[-2:]
                        norm = torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes.dtype)
                        if boxes.numel() > 0:
                            boxes = boxes / norm
                        detector_outputs.append({'boxes': boxes})
                # 2. Generate ignore regions
                ignore_regions = generate_ignore_regions(detector_outputs, targets, iou_threshold=0.5)
                # print(f"Generated ignore regions for detector_outputs {detector_outputs} \nand targets {targets} \n: {ignore_regions}")
            else:
                ignore_regions = None
            
            batch_losses, _ = model(images, targets, ignore_regions)

            loss = batch_losses['classification']
            loss += batch_losses['bbox_regression']

            # Check for NaN before combining losses
            if torch.isnan(batch_losses['classification']) or torch.isnan(batch_losses['bbox_regression']):
                print(f"NaN detected! Classification: {batch_losses['classification'].item()}, BBox: {batch_losses['bbox_regression'].item()}")
                print(f"Batch index: {idx}")
                print('Targets: {}'.format(targets))

            ssd_classification_losses.append(batch_losses['classification'].item())
            ssd_localization_losses.append(batch_losses['bbox_regression'].item())
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
        
        # Per-epoch intermediate mAP on dataset with config transform.
        epoch_eval_results_path = os.path.join(model_task_path, 'epoch_{:04d}_default_eval_results'.format(i + 1))
        epoch_eval_args = argparse.Namespace(**vars(args))
        epoch_eval_args.infer_samples = False
        epoch_eval_args.evaluate = True
        epoch_eval_args.eval_mode = 'default'
        epoch_eval_args.results_path = epoch_eval_results_path
        epoch_eval_result = infer_and_evaluate(epoch_eval_args)

        epoch_map = float('nan')
        epoch_recall = float('nan')
        if isinstance(epoch_eval_result, dict):
            evaluation = epoch_eval_result.get('evaluation', {})
            if isinstance(evaluation, dict):
                runs = evaluation.get('runs', [])
                if runs:
                    epoch_map = float(runs[0].get('mAP', float('nan')))
                    epoch_recall = float(runs[0].get('mean_detector_recall', float('nan')))
                else:
                    epoch_map = float(evaluation.get('mAP', float('nan')))
                    epoch_recall = float(evaluation.get('mean_detector_recall', float('nan')))

        metrics_csv_path = os.path.join(model_task_path, 'training_metrics.csv')
        append_epoch_metrics_csv(
            metrics_csv_path,
            epoch=i + 1,
            classification_loss=np.mean(ssd_classification_losses),
            detection_loss=np.mean(ssd_localization_losses),
            learning_rate=lr_scheduler.get_last_lr()[0],
            mean_ap=epoch_map,
            mean_detector_recall=epoch_recall,
        )
    print('Done Training...')
    print('Evaluating...')
    final_eval_args = argparse.Namespace(**vars(args))
    final_eval_args.infer_samples = True
    final_eval_args.evaluate = True
    final_eval_args.eval_mode = args.final_eval_mode
    final_eval_args.results_path = os.path.join(model_task_path, 'final_{}_results'.format(args.final_eval_mode))
    infer_and_evaluate(final_eval_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ssd training')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    parser.add_argument('--final-eval-mode', dest='final_eval_mode',
                        choices=['default', 'pad-loop'], default='pad-loop', type=str)
    args = parser.parse_args()
    train(args)
