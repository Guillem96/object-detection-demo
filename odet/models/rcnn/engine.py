import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn.metrics as metrics

import odet.utils.ss as ss
import odet.models.rcnn as rcnn


def _crop(im: torch.FloatTensor, 
          boxes: torch.LongTensor) -> torch.FloatTensor:
    def crop_single(box):
        return F.interpolate(
            im[:, box[1]: box[3], box[0]: box[2]].unsqueeze(0), 
            size=(224, 224))

    return torch.cat([crop_single(b) for b in boxes], dim=0)


def train_single_epoch(model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       loss_fn,
                       data_loader: torch.utils.data.DataLoader,
                       epoch: int,
                       device: torch.device):
    model.train()
    
    running_loss = 0.0
    running_reg_loss = 0.0
    running_clf_loss = 0.0

    n_steps = 0

    for i, (x, (boxes, reg_targets, labels)) in enumerate(data_loader):
        
        x = x.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        reg_targets = reg_targets.to(device)

        for batch in range(x.size(0)):
            model.zero_grad()
            optimizer.zero_grad()

            batch_labels = labels[batch]
            batch_boxes = boxes[batch].long()
            rt_batch = reg_targets[batch]

            batch_im = x[batch]
            
            # Crop the image in multiple regions
            crops = _crop(batch_im, batch_boxes)

            preds, regressions = model(crops)
            
            clf_loss, reg_loss = loss_fn(batch_labels, preds,
                                         rt_batch, regressions)
            
            if reg_loss == reg_loss: # not isnan
                loss = (clf_loss + reg_loss * .5)
            else: 
                loss = clf_loss * 0.5

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            
            optimizer.step()

            if reg_loss == reg_loss:
                running_reg_loss += reg_loss.item()
                running_loss += loss.item()
                running_clf_loss += clf_loss.item()
                n_steps += 1

        if (i + 1) % 10 == 0:
            loss_mean = running_loss / n_steps
            clf_loss_mean = running_clf_loss / n_steps
            reg_loss_mean = running_reg_loss / n_steps

            print(f'Epoch[{epoch}] [{i}/{len(data_loader)}]'
                  f'  loss: {loss_mean:.3f}'
                  f'  clf_loss: {clf_loss_mean:.3f}'
                  f'  reg_loss: {reg_loss_mean:.3f}')


@torch.no_grad()
def evaluate(model: nn.Module,
             loss_fn,
             data_loader: torch.utils.data.DataLoader,
             device: torch.device):
    model.eval()         
    
    running_loss = 0.0
    running_reg_loss = 0.0
    running_clf_loss = 0.0

    n_steps = 0

    true_labels = []
    pred_labels = []

    for i, (x, (boxes, reg_targets, labels)) in enumerate(data_loader):
        
        x = x.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        reg_targets = reg_targets.to(device)

        for batch in range(x.size(0)):

            batch_labels = labels[batch]
            rt_batch = reg_targets[batch]
            batch_boxes = boxes[batch].long()

            batch_im = x[batch]
            
            # Crop the image in multiple regions
            crops = _crop(batch_im, batch_boxes)

            preds, regressions = model(crops)
            clf_loss, reg_loss = loss_fn(batch_labels, preds,
                                         rt_batch, regressions)
            
            if reg_loss == reg_loss:
                loss = (clf_loss + reg_loss * .5)
                running_reg_loss += reg_loss.item()
                running_loss += loss.item()
                running_clf_loss += clf_loss.item()
                n_steps += 1

            true_labels.extend(batch_labels.cpu().numpy())
            pred_labels.extend(preds.argmax(-1).detach().cpu().numpy())

    loss_mean = running_loss / n_steps
    clf_loss_mean = running_clf_loss / n_steps
    reg_loss_mean = running_reg_loss / n_steps
    
    print(f'Evaluation '
          f'  loss: {loss_mean:.3f}'
          f'  clf_loss: {clf_loss_mean:.3f}'
          f'  reg_loss: {reg_loss_mean:.3f}')

    print(metrics.classification_report(true_labels, pred_labels))