import os
import random
import argparse
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torchvision
import torchvision.transforms as transforms

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from utils.prune_utils import *

from pytorch_nndct.apis import torch_quantizer, dump_xmodel




os.environ['CUDA_VISIBLE_DEVICES']='0'
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")




parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_dir',
    default="weights/slim_40",
    help='Trained model file path. Download pretrained model from the following url and put it in model_dir specified path')
parser.add_argument(
    '--data',
    type=str,
    default='data/coco.data',
    help='*.data file path')
parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
parser.add_argument(
    '--subset_len',
    default=None,
    type=int,
    help='subset_len to evaluate model, using the whole validation dataset if it is not set')
parser.add_argument(
    '--batch_size',
    default=16,
    type=int,
    help='input data batch size to evaluate model')
parser.add_argument('--fast_finetune', 
    dest='fast_finetune',
    action='store_true',
    help='fast finetune model before calibration')
parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')

args, _ = parser.parse_known_args()




hyp = {'giou': 1.582,  # giou loss gain
       'cls': 27.76,  # cls loss gain  (CE=~1.0, uCE=~20)
       'cls_pw': 1.446,  # cls BCELoss positive_weight
       'obj': 21.35,  # obj loss gain (*=80 for uBCE with 80 classes)
       'obj_pw': 3.941,  # obj BCELoss positive_weight
       'iou_t': 0.2635,  # iou training threshold
       'lr0': 0.002324,  # initial learning rate (SGD=1E-3, Adam=9E-5)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.97,  # SGD momentum
       'weight_decay': 0.0004569,  # optimizer weight decay
       'fl_gamma': 0.5,  # focal loss gamma
       'hsv_h': 0.10,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.5703,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.3174,  # image HSV-Value augmentation (fraction)
       'degrees': 1.113,  # image rotation (+/- deg)
       'translate': 0.06797,  # image translation (+/- fraction)
       'scale': 0.1059,  # image scale (+/- gain)
       'shear': 0.5768}  # image shear (+/- deg)




def load_data(train=True,
              data='data/coco.data',
              batch_size=16,
              subset_len=None,
              sample_method='random',
              distributed=False):

    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    val_path = data_dict['valid']
    train_sampler = None

    if train:
        dataset = LoadImagesAndLabels(path=train_path,
                                    batch_size=batch_size,
                                    augment=True,
                                    hyp=hyp,  # augmentation hyperparameters
                                    rect=False,  # rectangular training
                                    )
        if subset_len:
            assert subset_len <= len(dataset)
            if sample_method == 'random':
                dataset = torch.utils.datat.Subset(dataset, random.sample(range(0, len(dataset)), subset_len))
            else:
                dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                num_workers=min([os.cpu_count(), batch_size, 16]),
                                                shuffle=True,  # Shuffle=True unless rectangular training is used
                                                pin_memory=True,
                                                collate_fn=dataset.collate_fn)
    else:
        dataset = LoadImagesAndLabels(path=val_path, batch_size=batch_size)    
        if subset_len:
            assert subset_len <= len(dataset)
            if sample_method == 'random':
                dataset = torch.utils.datat.Subset(dataset, random.sample(range(0, len(dataset)), subset_len))
            else:
                dataset = torch.utils.data.Subset(dataset, list(range(subset_len)))
        data_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=min([os.cpu_count(), batch_size, 16]),
                                                 pin_memory=True,
                                                 collate_fn=dataset.collate_fn)
    return data_loader, train_sampler




def evaluate(model, data_loader):

    nc = 80  # number of classes, coco2017
    iou_thres=0.5
    conf_thres=0.001
    nms_thres=0.5

    seen = 0
    model.eval()
    #s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1')
    s = 'Running >w< !!! '
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    stats, ap, ap_class = [], [], []
    for _, (imgs, targets, paths, shapes) in enumerate(tqdm(data_loader, desc=s)):
        targets = targets.to(device)
        imgs = imgs.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Run model
        inf_out, train_out = model(imgs)  # inference and training outputs

        # Compute loss
        if hasattr(model, 'hyp'):  # if model has loss hyperparameters
            loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls

        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height

                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(m[bi])

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu().detach(), pred[:, 6].cpu().detach(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    p, r, ap, f1, ap_class = ap_per_class(*stats)
    mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss / len(data_loader)).tolist()), maps




def quantization(title='optimize', model_name=''):
    
    data = args.data
    quant_mode = args.quant_mode
    finetune = args.fast_finetune
    deploy = args.deploy
    batch_size = args.batch_size
    subset_len = args.subset_len
    model_dir = args.model_dir

    if quant_mode != 'test' and deploy:
        deploy = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    
    if deploy and (batch_size != 1 or subset_len != 1):
        print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
        batch_size = 1
        subset_len = 1

    # for .pt
    weight_path = os.path.join(model_dir, model_name + '.pt')
    # for .weights
    # weight_path = os.path.join(model_dir, model_name + '.weights')
    cfg_path = os.path.join(model_dir, model_name + '.cfg')
    img_size = 416


    model = Darknet(cfg_path, (img_size, img_size)).to(device)

    # for .pt
    model.load_state_dict(torch.load(weight_path, map_location=device)['model'])

    # for .weights
    # _ = load_darknet_weights(model, weight_path)

    input = torch.randn([batch_size, 3, 224, 224])

    if quant_mode == 'float':
        quant_model = model
    else:
        print('FIRST STEP')
        quantizer = torch_quantizer(quant_mode, model, (input), device=device)
        print('SECOND STEP')
        quant_model = quantizer.quant_model

    val_loader, _ = load_data(
        subset_len=subset_len,
        train=False,
        batch_size=batch_size,
        sample_method='random',
        data=data)

    # fast finetune model or load finetuned parameter before test
    if finetune == True:
        ft_loader, _ = load_data(
            subset_len=1024,
            train=False,
            batch_size=batch_size,
            sample_method=None,
            data=data)
        if quant_mode == 'calib':
            print('THIRD STEP')
            quantizer.fast_finetune(evaluate, (quant_model, ft_loader))
        elif quant_mode == 'test':
            quantizer.load_ft_param()
    
    # (mp, mr, map, mf1, *(loss / len(data_loader)).tolist()), maps
    results, _ = evaluate(quant_model, val_loader)

    print('Precision: %g' % (results[0]))
    print('Recall: %g' % (results[1]))
    print('mAP: %g' % (results[2]))
    print('F1: %g' % (results[3]))
    print('Loss: %g' % (results[4]))

    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if deploy:
        quantizer.export_xmodel(deploy_check=False)




if __name__ == '__main__':
    model_name = 'v4tiny_prune'
    feature_test = ' float model evaluation'
    
    if args.quant_mode != 'float':
        feature_test = ' quantization'
    else:
        feature_test = ' float model evaluation'
    title = model_name + feature_test

    print("-------- Start {} test ".format(model_name))

    quantization(title=title,model_name=model_name)
    
    print("-------- End of {} test ".format(model_name))
