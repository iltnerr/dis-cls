"""
Train and eval functions
"""
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import torch
import util.utils as utils

from .losses import DistillationLoss
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from timm.utils import accuracy, ModelEma
from typing import Iterable, Optional
from util.common import is_office


Image.MAX_IMAGE_PIXELS = None


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None,
                    set_training_mode=True):
    
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    for samples, targets in metric_logger.log_every(
            data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # with torch.cuda.amp.autocast():
        outputs = model(samples)
        loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # This attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        
        if not is_office(): # loss scaler requires a cuda device
            loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc3, acc5 = accuracy(output, target, topk=(1,3,5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc3'].update(acc3.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@3 {top3.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top3=metric_logger.acc3, top5=metric_logger.acc5, losses=metric_logger.loss))
    print(output.mean().item(), output.std().item())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_cm(net, test_loader, save_name, device, output_dir, class_indict):

    errors = 0
    y_pred, y_true = [], []
    net.load_state_dict(torch.load(save_name)['model'])

    net.eval()

    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        preds = torch.argmax(torch.softmax(net(images), dim=1), dim=1)
        for i in range(len(preds)):
            y_pred.append(preds[i].cpu())
            y_true.append(labels[i].cpu())

    tests = len(y_pred)

    for i in range(tests):
        pred_index = y_pred[i]
        true_index = y_true[i]
        if pred_index != true_index:
            errors += 1

    acc = (1 - errors / tests) * 100
    print(f'{errors} misclassifications in {tests} validation images: Acc@1 = {acc:6.2f}%')

    ypred = np.array(y_pred)
    ytrue = np.array(y_true)

    class_count = len(list(class_indict.values()))
    classes = list(class_indict.values())
    print("classes",classes)

    cm = confusion_matrix(ytrue, ypred, labels=[i for i in range(20)])
    pcm=np.zeros(shape=cm.shape)
    i=0
    j=0
    for i in range(20):
        for j in range(20):
            pcm[i][j]= cm[i][j]/np.sum(cm[i])
    plt.figure(figsize=(20, 20))
    sns.heatmap(pcm, annot=True, fmt='.1%')
    plt.xticks(np.arange(class_count)+ .5, classes, rotation=45, fontsize=12)
    plt.yticks(np.arange(class_count)+ .5, classes, rotation=0, fontsize=12)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.savefig(f'{output_dir}/confusion_matrix.pdf')

    clr = classification_report(y_true, y_pred, labels=[i for i in range(20)], target_names=classes, digits=4)
    print("Classification Report:\n----------------------\n", clr)

    with open(f'{output_dir}/ClassificationReport.txt', mode='a') as f:
        f.write( str(clr) + "\n")
