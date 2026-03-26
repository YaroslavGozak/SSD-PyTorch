default_model_config_params = {
                'aspect_ratios': [
                    [1., 2., 0.5],
                    [1., 2., 3., 0.5, .333],
                    [1., 2., 3., 0.5, .333],
                    [1., 2., 3., 0.5, .333],
                    [1., 2., 0.5],
                    [1., 2., 0.5]
                ],
                'scales': [0.1, 0.2, 0.375, 0.55, 0.725, 0.9],
                'iou_threshold': 0.5,
                'low_score_threshold': 0.01,
                'neg_pos_ratio': 3,
                'pre_nms_topK': 400,
                'detections_per_img': 200,
                'nms_threshold': 0.45,
            }