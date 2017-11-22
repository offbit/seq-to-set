import sys
import time

sys.path.append("/home/stathis/dev/spotlight/")
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.stats as st

from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.cross_validation import user_based_train_test_split
from spotlight.losses import hinge_loss

from nets import MatchNet
from data_loader import InteractionsSampler
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
    for i, (interactions, positive, negative, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        interactions_vb = Variable(interactions.cuda())
        target_vb = Variable(labels.cuda())
        predicted_vb = model(interactions_vb)

        loss = criterion(predicted_vb, target_vb)
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


def test_epoch(loader, model):
    losses = AverageMeter()

    # switch to train mode
    model.train(False)

    for i, (basket, positive, negative, _) in enumerate(loader):
        basket = basket.cuda()
        positive = positive.cuda()
        negative = negative.cuda()

        basket_vb = Variable(basket)
        positive_vb = Variable(positive)
        negative_vb = Variable(negative)

        # compute output
        # predicted = model(basket_vb, query_vb)
        user_representation = model.user_representation(basket_vb)

        positives = model(user_representation, positive_vb)
        negatives = model(user_representation, negative_vb)

        loss = hinge_loss(positives, negatives)

        losses.update(loss.data[0], loss.size(0))

    print('Test loss {}'.format(losses.avg))


def test_mrr(loader, model, num_items):
    # switch to train mode
    model.train(False)
    item_ids = np.arange(num_items).reshape(1, -1)
    item_ids = Variable(torch.from_numpy(item_ids.astype(np.int64)).cuda())
    mrrs = []

    prec = []
    recall = []
    for i, (basket, positive, negative, labels) in enumerate(loader):
        basket_vb = Variable(basket.cuda())
        user_representation = model.user_representation(basket_vb)
        # size = (num_items,) + user_representation.size()[1:]
        user_representation = user_representation[:, :, -1]
        user_representation = \
            user_representation.unsqueeze(2).repeat(1, 1, num_items)

        out = -model(user_representation, item_ids).cpu()
        out = out.data.numpy().flatten()
        positive = positive.numpy()
        next_item = positive[0, 0]
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

    max_sequence_length = 200
    min_sequence_length = 20
    step_size = 200
    random_state = np.random.RandomState(100)

    dataset = get_movielens_dataset('1M')

    train, rest = user_based_train_test_split(dataset,
                                              random_state=random_state)
    test, validation = user_based_train_test_split(rest,
                                                   test_percentage=0.5,
                                                   random_state=random_state)
    train = train.to_sequence(max_sequence_length=max_sequence_length,
                              min_sequence_length=min_sequence_length,
                              step_size=step_size)
    test = test.to_sequence(max_sequence_length=max_sequence_length,
                            min_sequence_length=min_sequence_length,
                            step_size=step_size)
    validation = validation.to_sequence(
        max_sequence_length=max_sequence_length,
        min_sequence_length=min_sequence_length,
        step_size=step_size)

    template_size = 40
    query_size = 10
    db_train = InteractionsSampler(train.sequences, train.sequence_lengths,
                                   template_size, query_size)
    db_test = InteractionsSampler(test.sequences, test.sequence_lengths,
                                  template_size, 10)
    model = MatchNet(train.num_items, embedding_dim=32,
                     item_embedding_layer=None, sparse=False,
                     nb_query=query_size)

    model.cuda()
    print(model)
    model.train()

    train_loader = DataLoader(db_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=True)

    parameters = model.parameters()
    optimizer = torch.optim.Adam(
        parameters,
        weight_decay=0.0002,
        lr=0.01
    )

    criterion = nn.MultiMarginLoss().cuda()

    start_epoch = 0
    epochs = 500
    for epoch in range(start_epoch, epochs):
        train_epoch(train_loader, model, criterion, optimizer, epoch)
        # if epoch % 5 == 0:
            # test_mrr(test_loader, model, test.num_items)
        # save_checkpoint(model)
