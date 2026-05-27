import os
import glob
import re
import csv
import cv2
import torch
import numpy as np
from PIL import Image
from collections import OrderedDict
from torchvision import transforms

from networks.SCDFNet import SCDFNet
from networks.modules.ASCJE import ASCJE
from path import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


src_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


def load_image(path):
    img = Image.open(path).convert("RGB")
    tensor = src_transform(img).unsqueeze(0)
    return img, tensor


def normalize_map(x):
    x = x.astype(np.float32)

    x_min = float(x.min())
    x_max = float(x.max())
    x_std = float(x.std())

    if x_max - x_min < 1e-8 or x_std < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)

    low = np.percentile(x, 1)
    high = np.percentile(x, 99)

    if high - low < 1e-8:
        low = x_min
        high = x_max

    x = np.clip(x, low, high)
    x = x - x.min()
    x = x / (x.max() + 1e-8)

    x = np.power(x, 0.6)

    x = (x * 255).astype(np.uint8)
    return x


def tensor_to_activation_map(tensor, mode="abs_mean"):
    x = tensor.detach()[0]  # [C, H, W]

    if mode == "abs_mean":
        act = x.abs().mean(dim=0)
    elif mode == "abs_max":
        act = x.abs().max(dim=0)[0]
    elif mode == "mean":
        act = x.mean(dim=0)
    else:
        raise ValueError("Unknown mode: {}".format(mode))

    return act.cpu().numpy()


def save_heatmap_from_map(act_map, raw_img, save_path, alpha=0.55):
    act = normalize_map(act_map)

    raw_img = np.array(raw_img)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)

    act = cv2.resize(act, (raw_img.shape[1], raw_img.shape[0]))

    heatmap = cv2.applyColorMap(act, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(raw_img, 1 - alpha, heatmap, alpha, 0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, overlay)


def save_pure_heatmap_from_map(act_map, raw_img, save_path):
    act = normalize_map(act_map)

    raw_img = np.array(raw_img)
    h, w = raw_img.shape[:2]

    act = cv2.resize(act, (w, h))
    heatmap = cv2.applyColorMap(act, cv2.COLORMAP_JET)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, heatmap)


def save_gray_map_from_map(act_map, raw_img, save_path):
    act = normalize_map(act_map)

    raw_img = np.array(raw_img)
    h, w = raw_img.shape[:2]

    act = cv2.resize(act, (w, h))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, act)


def get_all_ascje_modules(model):
    ascje_list = []

    for name, module in model.named_modules():
        if isinstance(module, ASCJE):
            ascje_list.append((name, module))

    return ascje_list


def load_checkpoint_safely(model, model_path, device):
    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    model_state = model.state_dict()
    new_state_dict = OrderedDict()

    unused_keys = []
    shape_mismatch_keys = []

    for k, v in ckpt.items():
        old_k = k

        k = k.replace("module.", "")

        if "total_ops" in k or "total_params" in k:
            continue

        k = re.sub(r"^(Conv[2-5]_[12])\.conv\.", r"\1.Conv.", k)
        k = re.sub(r"^(UpFuse[2-5]\.refine)\.conv\.", r"\1.Conv.", k)

        if k in model_state:
            if model_state[k].shape == v.shape:
                new_state_dict[k] = v
            else:
                shape_mismatch_keys.append(
                    (old_k, k, tuple(v.shape), tuple(model_state[k].shape))
                )
        else:
            unused_keys.append(old_k)

    missing_keys = [k for k in model_state.keys() if k not in new_state_dict]

    print("=" * 80)
    print("Checkpoint loading information")
    print("Matched keys:", len(new_state_dict))
    print("Missing keys:", len(missing_keys))
    print("Unused checkpoint keys:", len(unused_keys))
    print("Shape mismatch keys:", len(shape_mismatch_keys))
    print("=" * 80)

    if len(missing_keys) > 0:
        print("\nFirst 30 missing keys:")
        for k in missing_keys[:30]:
            print("  ", k)

    if len(unused_keys) > 0:
        print("\nFirst 30 unused checkpoint keys:")
        for k in unused_keys[:30]:
            print("  ", k)

    if len(shape_mismatch_keys) > 0:
        print("\nFirst 30 shape mismatch keys:")
        for old_k, new_k, ckpt_shape, model_shape in shape_mismatch_keys[:30]:
            print("  {} -> {}: ckpt {} != model {}".format(
                old_k, new_k, ckpt_shape, model_shape
            ))

    model.load_state_dict(new_state_dict, strict=True)
    return model


def main():
    print("CUDA:", torch.cuda.is_available())

    model = SCDFNet(3, 1, ratio=0.5).to(device)

    model_path = ("ckps/SCDFNet_CDD/"
                  "SCDFNet_CDD_batch=32_lr=0.0001_epoch197model.pth")

    model = load_checkpoint_safely(model, model_path, device)
    model.eval()

    ascje_modules = get_all_ascje_modules(model)

    if len(ascje_modules) == 0:
        raise RuntimeError("当前模型中没有找到 ASCJE 模块。")

    print("\nAll ASCJE modules:")
    for i, (name, _) in enumerate(ascje_modules):
        print(i, name)

    target_files = sorted([
        os.path.basename(p)
        for p in glob.glob(os.path.join(test_src_t1, "*.png"))
    ])

    if len(target_files) == 0:
        raise RuntimeError(
            "没有在 {} 中找到 png 图像，请检查 path.py 中 test_src_t1 是否正确。".format(test_src_t1)
        )

    print("\nTotal test images:", len(target_files))

    save_root = os.path.join(test_root, "activation_heatmap_feature")
    os.makedirs(save_root, exist_ok=True)

    stats_path = os.path.join(save_root, "activation_stats.csv")

    selected_layer_indices = list(range(len(ascje_modules)))

    activation_mode = "abs_mean"

    stats_rows = []

    with torch.no_grad():
        for idx, file_name in enumerate(target_files):
            t1_path = os.path.join(test_src_t1, file_name)
            t2_path = os.path.join(test_src_t2, file_name)

            if not os.path.exists(t2_path):
                print("[Skip] t2 image not found:", t2_path)
                continue

            raw_t1, x1 = load_image(t1_path)
            raw_t2, x2 = load_image(t2_path)

            x1 = x1.to(device)
            x2 = x2.to(device)

            pred = model(x1, x2)

            pred_map = pred.detach()[0, 0].cpu().numpy()
            pred_dir = os.path.join(save_root, "prediction_prob_map")
            pred_overlay_path = os.path.join(
                pred_dir,
                file_name.replace(".png", "_pred_overlay.png")
            )
            pred_pure_path = os.path.join(
                pred_dir,
                file_name.replace(".png", "_pred_pure.png")
            )

            save_heatmap_from_map(pred_map, raw_t2, pred_overlay_path, alpha=0.55)
            save_pure_heatmap_from_map(pred_map, raw_t2, pred_pure_path)

            for layer_idx in selected_layer_indices:
                layer_name, layer_module = ascje_modules[layer_idx]

                if not hasattr(layer_module, "last_out"):
                    print("[Skip] no last_out:", layer_name)
                    continue

                feat = layer_module.last_out

                act_map = tensor_to_activation_map(
                    feat,
                    mode=activation_mode
                )

                act_min = float(act_map.min())
                act_max = float(act_map.max())
                act_mean = float(act_map.mean())
                act_std = float(act_map.std())

                stats_rows.append([
                    file_name,
                    layer_idx,
                    layer_name,
                    activation_mode,
                    act_min,
                    act_max,
                    act_mean,
                    act_std
                ])

                safe_layer_name = layer_name.replace(".", "_")

                layer_save_dir = os.path.join(
                    save_root,
                    "layer_{:02d}_{}".format(layer_idx, safe_layer_name)
                )

                overlay_save_path = os.path.join(
                    layer_save_dir,
                    file_name.replace(".png", "_feature_overlay.png")
                )

                pure_save_path = os.path.join(
                    layer_save_dir,
                    file_name.replace(".png", "_feature_pure.png")
                )

                gray_save_path = os.path.join(
                    layer_save_dir,
                    file_name.replace(".png", "_feature_gray.png")
                )

                save_heatmap_from_map(
                    act_map=act_map,
                    raw_img=raw_t2,
                    save_path=overlay_save_path,
                    alpha=0.55
                )

                save_pure_heatmap_from_map(
                    act_map=act_map,
                    raw_img=raw_t2,
                    save_path=pure_save_path
                )

                save_gray_map_from_map(
                    act_map=act_map,
                    raw_img=raw_t2,
                    save_path=gray_save_path
                )

            print("[{}/{}] Saved feature activation maps for: {}".format(
                idx + 1, len(target_files), file_name
            ))

    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_name",
            "layer_idx",
            "layer_name",
            "activation_mode",
            "min",
            "max",
            "mean",
            "std"
        ])
        writer.writerows(stats_rows)

    print("\nAll feature activation heatmaps saved to:", save_root)
    print("Activation statistics saved to:", stats_path)


if __name__ == "__main__":
    main()