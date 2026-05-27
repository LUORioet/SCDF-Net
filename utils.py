import logging
import os
import torch
from PIL import Image
import numpy as np


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    log_dir = os.path.dirname(filename)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger_name = name if name is not None else os.path.splitext(os.path.basename(filename))[0]
    logger = logging.getLogger(logger_name)
    logger.setLevel(level_dict[verbosity])
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )

    fh = logging.FileHandler(filename, "w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def count_parameters_m(model):
    params = sum(p.numel() for p in model.parameters())
    return params / 1e6


def compute_model_complexity(model, input_size=(3, 256, 256), device=None):
    if device is None:
        device = next(model.parameters()).device

    params_m = count_parameters_m(model)
    flops_g = None
    error_msg = None

    try:
        from thop import profile

        was_training = model.training
        model.eval()

        dummy_t1 = torch.randn(1, *input_size).to(device)
        dummy_t2 = torch.randn(1, *input_size).to(device)

        with torch.no_grad():
            flops, _ = profile(model, inputs=(dummy_t1, dummy_t2), verbose=False)

        flops_g = flops / 1e9
        model.train(was_training)

    except Exception as e:
        error_msg = str(e)

    return params_m, flops_g, error_msg


def log_model_complexity(logger, model, input_size=(3, 256, 256), device=None):
    params_m, flops_g, error_msg = compute_model_complexity(
        model=model,
        input_size=input_size,
        device=device
    )

    logger.info("=" * 70)
    logger.info("Model Complexity")
    logger.info("Input Size: 1 x {} x {} x {}".format(input_size[0], input_size[1], input_size[2]))
    logger.info("Params (M): {:.4f}".format(params_m))

    if flops_g is None:
        logger.info("FLOPs  (G): N/A")
        logger.info("FLOPs Error: {}".format(error_msg))
        logger.info("Tip: install thop first, e.g. pip install thop")
    else:
        logger.info("FLOPs  (G): {:.4f}".format(flops_g))

    logger.info("=" * 70)

    return params_m, flops_g


def _mask_tensor_to_uint8(mask, threshold=0.5):
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu()

    if mask.dim() == 4:
        mask = mask[0, 0]
    elif mask.dim() == 3:
        mask = mask[0]

    mask = mask.float()
    mask = (mask >= threshold).numpy().astype(np.uint8) * 255
    return mask


# def save_pre_result(pre, save_path, file_name, threshold=0.5, prefix="predict_"):
#     '''predict'''
#     os.makedirs(save_path, exist_ok=True)
#
#     outputs = _mask_tensor_to_uint8(pre, threshold=threshold)
#     outputs = Image.fromarray(outputs)
#
#     save_name = prefix + file_name
#     outputs.save(os.path.join(save_path, save_name))


def save_pre_result(pre, flag, num, save_path, threshold=0.5):
    '''norm'''
    os.makedirs(save_path, exist_ok=True)
    outputs = _mask_tensor_to_uint8(pre, threshold=threshold)
    outputs = Image.fromarray(outputs)
    outputs.save(os.path.join(save_path, "{}_{}.png".format(flag, num)))
