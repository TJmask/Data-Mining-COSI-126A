from __future__ import division
import argparse
import torch
from torch.utils import model_zoo
from torch.autograd import Variable

import models
import utils
from data_loader import get_train_test_loader, get_office31_dataloader


CUDA = True if torch.cuda.is_available() else False
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
BATCH_SIZE = [200, 56]
EPOCHS = 16


source_loader = get_office31_dataloader(case='dslr', batch_size=BATCH_SIZE[0])
target_loader = get_office31_dataloader(case='amazon', batch_size=BATCH_SIZE[1])


def train(model, optimizer, epoch, _lambda):
    result = []

    # Expected size : xs -> (batch_size, 3, 300, 300), ys -> (batch_size)
    source, target = list(enumerate(source_loader)), list(enumerate(target_loader))
    train_steps = min(len(source), len(target))

    for batch_idx in range(train_steps):
        _, (source_data, source_label) = source[batch_idx]
        _, (target_data, _) = target[batch_idx]
        if CUDA:
            source_data = source_data.cuda()
            source_label = source_label.cuda()
            target_data = target_data.cuda()

        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)

        optimizer.zero_grad()
        out1, out2 = model(source_data, target_data)

        classification_loss = torch.nn.functional.cross_entropy(out1, source_label)
        coral_loss = models.CORAL(out1, out2)

        sum_loss = _lambda*coral_loss + classification_loss
        sum_loss.backward()

        optimizer.step()

        result.append({
            'epoch': epoch,
            'step': batch_idx + 1,
            'total_steps': train_steps,
            'lambda': _lambda,
            'coral_loss': coral_loss.data,
            'classification_loss': classification_loss.data,
            'total_loss': sum_loss.data
        })

        print('Train Epoch: {:2d} [{:2d}/{:2d}]\t'
              'Lambda: {:.4f}, Class: {:.6f}, CORAL: {:.6f}, Total_Loss: {:.6f}'.format(
                  epoch,
                  batch_idx + 1,
                  train_steps,
                  _lambda,
                  classification_loss.data,
                  coral_loss.data,
                  sum_loss.data
              ))

    return result


def test(model, dataset_loader, e):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in dataset_loader:
        if CUDA:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data, volatile=True), Variable(target)
        out, _ = model(data, data)

        # sum up batch loss
        test_loss += torch.nn.functional.cross_entropy(out, target, size_average=False).data

        # get the index of the max log-probability
        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(dataset_loader.dataset)

    return {
        'epoch': e,
        'average_loss': test_loss,
        'correct': correct,
        'total': len(dataset_loader.dataset),
        'accuracy': 100. * correct / len(dataset_loader.dataset)
    }


# load AlexNet pre-trained model
def load_pretrained(model):
    url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    pretrained_dict = model_zoo.load_url(url)
    model_dict = model.state_dict()

    # filter out unmatch dict and delete last fc bias, weight
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # del pretrained_dict['classifier.6.bias']
    # del pretrained_dict['classifier.6.weight']

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='Resume from checkpoint file')
    args = parser.parse_args()

    model = models.DeepCORAL(31)

    # support different learning rate according to CORAL paper
    # i.e. 10 times learning rate for the last two fc layers.
    optimizer = torch.optim.SGD([
        {'params': model.sharedNet.parameters()},
        {'params': model.fc.parameters(), 'lr': 10*LEARNING_RATE},
    ], lr=LEARNING_RATE, momentum=MOMENTUM)

    if CUDA:
        model = model.cuda()

    if args.load is not None:
        utils.load_net(model, args.load)
    else:
        load_pretrained(model.sharedNet)

    training_statistic = []
    testing_s_statistic = []
    testing_t_statistic = []

    for e in range(0, EPOCHS):
        _lambda = (e+1)/EPOCHS
        # _lambda = 0.0
        res = train(model, optimizer, e+1, _lambda)
        print('###EPOCH {}: Class: {:.6f}, CORAL: {:.6f}, Total_Loss: {:.6f}'.format(
            e+1,
            sum(row['classification_loss'] / row['total_steps'] for row in res),
            sum(row['coral_loss'] / row['total_steps'] for row in res),
            sum(row['total_loss'] / row['total_steps'] for row in res),
        ))

        training_statistic.append(res)

        test_source = test(model, source_loader, e)
        test_target = test(model, target_loader, e)
        testing_s_statistic.append(test_source)
        testing_t_statistic.append(test_target)

        print('###Test Source: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            e+1,
            test_source['average_loss'],
            test_source['correct'],
            test_source['total'],
            test_source['accuracy'],
        ))
        print('###Test Target: Epoch: {}, avg_loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            e+1,
            test_target['average_loss'],
            test_target['correct'],
            test_target['total'],
            test_target['accuracy'],
        ))

    torch.save(training_statistic, 'training_statistic.pkl')
    torch.save(testing_s_statistic, 'testing_s_statistic.pkl')
    torch.save(testing_t_statistic, 'testing_t_statistic.pkl')
    torch.save_net(model, 'checkpoint.tar')
