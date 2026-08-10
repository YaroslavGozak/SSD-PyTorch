import sys
from dataset.imagenet_vid import ImageNetVidDataset

try:
    print("Testing ImageNetVidDataset with directory walking...")
    
    # Use val (test) split which has simpler structure
    dataset = ImageNetVidDataset(
        split='test',
        train_data_root=r"D:\ImageNet-VID\ImageNet\data\ImageNet2015\object_detection_from_video\ILSVRC2015\Data\VID\train",
        train_ann_root=r"D:\ImageNet-VID\ImageNet\data\ImageNet2015\object_detection_from_video\ILSVRC2015\Annotations\VID\train",
        test_data_root=r"D:\ImageNet-VID\ImageNet\data\ImageNet2015\object_detection_from_video\ILSVRC2015\Data\VID\val",
        test_ann_root=r"D:\ImageNet-VID\ImageNet\data\ImageNet2015\object_detection_from_video\ILSVRC2015\Annotations\VID\val",
        im_size=300,
        transform_name='ssd',
        task='demo'
    )
    
    print(f"✓ Dataset initialized successfully")
    print(f"✓ Dataset length: {len(dataset)}")
    print(f"✓ Number of classes: {len(dataset.label2idx)}")
    
    if len(dataset) > 0:
        print("\nLoading first sample...")
        im_tensor, targets, filename = dataset[0]
        print(f"  ✓ Image shape: {im_tensor.shape}")
        print(f"  ✓ Target keys: {list(targets.keys())}")
        print(f"  ✓ Video metadata: video_id={targets['video_id']}, frame_idx={targets['frame_idx']}, is_first={targets['is_first_frame']}")
        print("\n✓✓ All tests passed!")
    else:
        print("✗ No frames loaded")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
