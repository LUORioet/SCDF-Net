import os
import re
from collections import OrderedDict
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from operation import predict
from path import *
import torch
from dataset import RsDataset
from networks.SCDFNet import SCDFNet


# norm

print('CUDA: ', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]

src_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(512),
    transforms.Normalize([0.5], [0.5])
])

label_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(512),
])

model = SCDFNet(3, 1, ratio=0.5).to(device)
# model = torch.nn.DataParallel(model, device_ids=device_ids).to(device)
model_path = ('ckps/x/'
              'y.pth')
ckps = torch.load(model_path, map_location='cuda:0')
model.load_state_dict(ckps)

dataset_test = RsDataset(test_src_t1, test_src_t2, test_label, test=True,
                         t1_transform=src_transform,
                         t2_transform=src_transform,
                         label_transform=label_transform)

dataloader_test = DataLoader(dataset_test,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4)

pre_test, rec_test, f1_test, iou_test, kc_test = predict(model, dataloader_test)
print('test Pre:(%f,%f) test Recall:(%f,%f) test MeanF1Score:(%f,%f) test IoU:(%f,%f) test KC: %f' % (
    pre_test['precision_0'], pre_test['precision_1'], rec_test['recall_0'], rec_test['recall_1'], f1_test['f1_0'],
    f1_test['f1_1'], iou_test['iou_0'], iou_test['iou_1'], kc_test))




'''
#predict

print('CUDA: ', torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

src_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

label_transform = transforms.Compose([
    transforms.ToTensor(),
])


def load_checkpoint_safely(model, model_path, device):
    ckps = torch.load(model_path, map_location=device)

    if isinstance(ckps, dict) and "state_dict" in ckps:
        ckps = ckps["state_dict"]

    model_state = model.state_dict()
    new_state_dict = OrderedDict()

    unused_keys = []

    for k, v in ckps.items():
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
                print("[Shape mismatch] {}: ckpt {} != model {}".format(
                    k, tuple(v.shape), tuple(model_state[k].shape)
                ))
        else:
            unused_keys.append(old_k)

    missing_keys = [k for k in model_state.keys() if k not in new_state_dict]

    print("Matched keys:", len(new_state_dict))
    print("Missing keys:", len(missing_keys))
    print("Unused checkpoint keys:", len(unused_keys))

    if len(missing_keys) > 0:
        print("\nFirst 30 missing keys:")
        for k in missing_keys[:30]:
            print("  ", k)

    if len(unused_keys) > 0:
        print("\nFirst 30 unused checkpoint keys:")
        for k in unused_keys[:30]:
            print("  ", k)

    model.load_state_dict(new_state_dict, strict=True)
    return model

model = SCDFNet(3, 1, ratio=0.5).to(device)
model_path = (
    'ckps/SCDFNet_DSIFN/'
    'SCDFNet_DSIFN_batch=32_lr=0.0001_epoch146model.pth'
)
model = load_checkpoint_safely(model, model_path, device)

target_files = sorted([
    os.path.basename(p)
    for p in glob.glob(os.path.join(test_src_t1, "*.png"))
])

dataset_test = RsDataset(test_src_t1,test_src_t2,test_label,test=True,
                         target_files=target_files,
                         t1_transform=src_transform,
                         t2_transform=src_transform,
                         label_transform=label_transform
                         )


dataloader_test = DataLoader(dataset_test,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4)

pre_test, rec_test, f1_test, iou_test, kc_test = predict(model, dataloader_test)
print('test Pre:(%f,%f) test Recall:(%f,%f) test MeanF1Score:(%f,%f) test IoU:(%f,%f) test KC: %f' % (
    pre_test['precision_0'], pre_test['precision_1'], rec_test['recall_0'], rec_test['recall_1'], f1_test['f1_0'],
    f1_test['f1_1'], iou_test['iou_0'], iou_test['iou_1'], kc_test))

'''