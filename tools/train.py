from tools.infer import infer_and_evaluate
import torch
import argparse
import os
import numpy as np
import yaml
import random
import csv
import torchvision
from tqdm import tqdm
from dataset.visdrone import VisDroneDataset
from dataset.visdroneroissd import VisDroneRoiSsdDataset
from dataset.voc import VOCDataset
from dataset.ytbb import YTBBDataset
from model.roissd import RoiSSD, generate_ignore_regions
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


def collate_function(data):
    return tuple(zip(*data))


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    #########################

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    if str(train_config['dataset']) == 'vis-drone':
        dataset = VisDroneDataset('train',
                     im_sets=dataset_config['train_im_sets'],
                     im_size=dataset_config['im_size'])
    elif str(train_config['dataset']) == 'ytbb':
        dataset = YTBBDataset('train',
                     root_dir=dataset_config['root_dir'],
                     im_size=dataset_config['im_size'])
    elif str(train_config['dataset']) == 'voc':
        dataset = VOCDataset('train',
                     im_sets=dataset_config['train_im_sets'],
                     im_size=dataset_config['im_size'],
                     transform_name=dataset_config['transform_name'])
    else:
        raise Exception('Unknown dataset name {}'.format(train_config['dataset']))
    train_dataset_loader = DataLoader(dataset,
                               batch_size=train_config['batch_size'],
                               shuffle=True,
                               collate_fn=collate_function,
                            #    num_workers=4,  # 0 - 1 process, 4 or 8 - number of processes
                            #    pin_memory=True,  # Add this for faster GPU transfer
                            #    persistent_workers=True, # Keep workers alive between epochs
                            #    prefetch_factor=2  # Prefetch 2 batches per worker
                               ) 

    # Instantiate model and load checkpoint if present
    if str(train_config['model']) == 'ssd':
        model = SSD(config=config['model_params'],
                num_classes=dataset_config['num_classes'])
    elif str(train_config['model']) == 'roissd':
        model = RoiSSD(config=config['model_params'],
                num_classes=dataset_config['num_classes'])
    else:
        raise Exception('Unknown model name {}'.format(train_config['model']))
    
    pretrained_detector = torchvision.models.detection.ssd300_vgg16(weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT)
    pretrained_detector.to(device)
    pretrained_detector.eval()
    
    model.to(device)

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
                writer.writerow(['epoch', 'classification_loss', 'detection_loss'])
            
            # Write current epoch data
            writer.writerow([i+1, np.mean(ssd_classification_losses), np.mean(ssd_localization_losses)])
    print('Done Training...')
    print('Evaluating...')
    args.infer_samples = True
    args.evaluate = True
    args.samples_path = os.path.join(model_task_path, 'samples')
    infer_and_evaluate(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ssd training')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    args = parser.parse_args()
    train(args)
