import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.stats as st
import torch.nn.functional as F

from nets import SeqModel
from data_loader import InteractionsSampler
from data_loader import read_dataset
from helpers import AverageMeter
from helpers import get_precision_recall


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    # filename = args.name + '.pth.tar'
    torch.save(state, filename)


def train_epoch(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    ntokens = model.num_items
    for i, (basket, positive, negative, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = basket[:, 1:].cuda()
        basket = basket[:, :-1].cuda()

        basket_vb = Variable(basket)
        target_vb = Variable(target)

        predicted_vb = model(basket_vb)
        # predicted_vb = F.log_softmax(predicted_vb)
        loss = criterion(predicted_vb.view(-1, ntokens), target_vb.view(-1))
        losses.update(loss.data[0], loss.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # we want clipping ?
        torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))


def test_mrr(loader, model, num_items):
    # switch to train mode
    model.train(False)
    mrrs = []

    prec = []
    recall = []
    for i, (basket, positive, negative, labels) in enumerate(loader):
        basket_vb = Variable(basket.cuda())
        predictions = model(basket_vb)[:,-1]  # pick last prediction
        out = -predictions.cpu()
        out = out.data.numpy().flatten()
        positive = positive.numpy()
        next_item = positive[0, 1] # pick next item (because idx:0 == BOS)
        # compute mrr
        mrr1 = (1.0 / st.rankdata(out)[next_item]).mean()
        mrrs.append(mrr1)
        out = out.argsort()
        prec_, recall_ = get_precision_recall(out, positive.flatten(), 10)
        prec.append(prec_)
        recall.append(recall_)

    print('Mrr:{}'.format(np.mean(mrrs)))
    print('mean prec: {} , mean recall: {}'.format(np.mean(prec),
                                                   np.mean(recall)))


if __name__ == '__main__':

    train, test, _ = read_dataset()

    template_size = 70
    query_size = 2
    db_train = InteractionsSampler(train.sequences, train.sequence_lengths,
                                   template_size, query_size)
    db_test = InteractionsSampler(test.sequences, test.sequence_lengths,
                                  template_size, 10)
    model = SeqModel(db_train.num_items, embedding_dim=32)
    # model = torch.load('checkpoint.pth.tar')
    model.cuda()
    print(model)
    model.train()

    train_loader = DataLoader(db_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=True)

    parameters = model.parameters()
    optimizer = torch.optim.Adam(
        parameters,
        weight_decay=0.0002,
        lr=0.001
    )

    criterion = nn.CrossEntropyLoss().cuda()

    start_epoch = 0
    epochs = 500
    for epoch in range(start_epoch, epochs):
        train_epoch(train_loader, model, criterion, optimizer, epoch)
        if epoch % 5 == 0:
            test_mrr(test_loader, model, db_train.num_items)
        save_checkpoint(model)
