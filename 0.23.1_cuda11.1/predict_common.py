import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcls.datasets.pipelines import Compose


def inference_model(model, img, top_k=None):
    """
    Inference image(s) with the classifier.

    Based on mmcls.apis.inference module.

    :param model: The loaded classifier.
    :type model: nn.Module
    :param img: the image filename or loaded image)
    :type img: str or np.ndarray
    :param top_k: whether to return just the top K predictions or all (when None)
    :type top_k: int
    :return: the dictionary with with class labels and their associated scores
    :rtype: dict
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    result = dict()
    with torch.no_grad():
        scores = model(return_loss=False, **data)
        if top_k is not None:
            sorted = np.flip(np.argsort(scores, axis=1))
            for k in range(top_k):
                i = sorted[0][k]
                result[model.CLASSES[i]] = float(scores[0][i])
        else:
            for i in range(len(scores[0])):
                result[model.CLASSES[i]] = float(scores[0][i])
    return result
