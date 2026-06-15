from dataset.helpers.label_spaces import (
    COCO_CLASSES,
    COCO_TO_IMAGENET_VID_NAME,
    IMAGENET_VID_CLASSES,
    IMAGENET_VID_YOLO13_CLASSES,
    IMAGENET_VID_YOLO13_TO_IMAGENET_VID_NAME,
    IMAGENET_VID_TO_VOC_NAME,
    VOC_CLASSES,
    VOC_TO_IMAGENET_VID_NAME,
    build_label_maps,
    get_label_space_num_classes,
    infer_dataset_label_space,
)
from model.model_adapters import DetectionLabelRemapAdapter


VOC_LABEL2IDX, VOC_IDX2LABEL = build_label_maps(VOC_CLASSES)
COCO_LABEL2IDX, COCO_IDX2LABEL = build_label_maps(COCO_CLASSES)
IMAGENET_VID_LABEL2IDX, IMAGENET_VID_IDX2LABEL = build_label_maps(IMAGENET_VID_CLASSES)
IMAGENET_VID_YOLO13_LABEL2IDX, IMAGENET_VID_YOLO13_IDX2LABEL = build_label_maps(IMAGENET_VID_YOLO13_CLASSES)


def get_model_label_space(train_config, dataset_name: str) -> str:
    return str(train_config.get('model_label_space', infer_dataset_label_space(dataset_name)))


def get_model_num_classes(train_config, dataset_config, dataset_name: str) -> int:
    if 'model_num_classes' in train_config:
        print('Using model_num_classes from train_config:', train_config['model_num_classes'])
        return int(train_config['model_num_classes'])
    print(train_config)

    model_label_space = get_model_label_space(train_config, dataset_name)
    dataset_label_space = infer_dataset_label_space(dataset_name)
    if model_label_space != dataset_label_space:
        return get_label_space_num_classes(model_label_space)
    return int(dataset_config['num_classes'])


def should_filter_imagenet_vid_to_voc_overlap(train_config, dataset_config, dataset_name: str) -> bool:
    if str(dataset_name) != 'imagenet-vid':
        return False
    if 'filter_voc_overlap' in dataset_config:
        return bool(dataset_config['filter_voc_overlap'])
    return get_model_label_space(train_config, dataset_name) == 'voc'


def maybe_wrap_model_for_dataset(model, dataset, train_config, dataset_name: str):
    dataset_label_space = infer_dataset_label_space(dataset_name)
    model_label_space = get_model_label_space(train_config, dataset_name)
    if {
        str(model_label_space),
        str(dataset_label_space),
    } <= {"imagenet-vid", "yolo-imagenet-vid"}:
        print(
            "No label remapping needed "
            f"(model_label_space={model_label_space}, dataset_label_space={dataset_label_space})"
        )
        return model

    if model_label_space == dataset_label_space:
        print(f"No label remapping needed (model_label_space={model_label_space}, dataset_label_space={dataset_label_space})")
        return model

    if (model_label_space, dataset_label_space) == ('voc', 'imagenet-vid'):
        print(f"Wrapping model with VOC->VID label remap adapter (model_label_space={model_label_space}, dataset_label_space={dataset_label_space})")
        return DetectionLabelRemapAdapter(
            base_model=model,
            source_idx2label=VOC_IDX2LABEL,
            target_label2idx=dataset.label2idx,
            class_name_mapping=VOC_TO_IMAGENET_VID_NAME,
        )
    if (model_label_space, dataset_label_space) == ('coco', 'imagenet-vid'):
        print(f"Wrapping model with COCO->VID label remap adapter (model_label_space={model_label_space}, dataset_label_space={dataset_label_space})")
        return DetectionLabelRemapAdapter(
            base_model=model,
            source_idx2label=COCO_IDX2LABEL,
            target_label2idx=dataset.label2idx,
            class_name_mapping=COCO_TO_IMAGENET_VID_NAME,
        )
    if (model_label_space, dataset_label_space) == ('imagenet-vid-yolo13', 'imagenet-vid'):
        print(
            f"Wrapping model with YOLO13->VID label remap adapter "
            f"(model_label_space={model_label_space}, dataset_label_space={dataset_label_space})"
        )
        return DetectionLabelRemapAdapter(
            base_model=model,
            source_idx2label=IMAGENET_VID_YOLO13_IDX2LABEL,
            target_label2idx=dataset.label2idx,
            class_name_mapping=IMAGENET_VID_YOLO13_TO_IMAGENET_VID_NAME,
        )
    if (model_label_space, dataset_label_space) == ('imagenet-vid', 'voc'):
        print(f"Wrapping model with VID->VOC label remap adapter (model_label_space={model_label_space}, dataset_label_space={dataset_label_space})")
        return DetectionLabelRemapAdapter(
            base_model=model,
            source_idx2label=IMAGENET_VID_IDX2LABEL,
            target_label2idx=dataset.label2idx,
            class_name_mapping=IMAGENET_VID_TO_VOC_NAME,
        )
    if (model_label_space, dataset_label_space) == ('yolo-imagenet-vid', 'voc'):
        print(
            "Wrapping model with YOLO-VID->VOC label remap adapter "
            f"(model_label_space={model_label_space}, dataset_label_space={dataset_label_space})"
        )
        return DetectionLabelRemapAdapter(
            base_model=model,
            source_idx2label=IMAGENET_VID_IDX2LABEL,
            target_label2idx=dataset.label2idx,
            class_name_mapping=IMAGENET_VID_TO_VOC_NAME,
        )

    raise ValueError(
        f'Unsupported model/dataset label-space combination: {model_label_space!r} -> {dataset_label_space!r}'
    )
