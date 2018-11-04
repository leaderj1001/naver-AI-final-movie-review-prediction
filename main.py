# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import math

from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

import nsml
from dataset import MovieReviewDataset, preprocess
from nsml import DATASET_PATH, HAS_DATASET, GPU_NUM, IS_ON_NSML

# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(model, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(filename, *args):
        checkpoint = {
            'model': model.state_dict()
        }
        torch.save(checkpoint, filename)

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(filename, *args):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """
        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """

        #print(raw_data)

        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)

        model.eval()
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        output_prediction = model(preprocessed_data)
        point = output_prediction.data.squeeze(dim=1).tolist()
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(confidence interval, 포인트)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 confidence interval의 값은 영향을 미치지 않습니다
        return list(zip(np.zeros(len(point)), point))

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)

def collate_fn(data: list):
    """
    PyTorch DataLoader에서 사용하는 collate_fn 입니다.
    기본 collate_fn가 리스트를 flatten하기 때문에 벡터 입력에 대해서 사용이 불가능해, 직접 작성합니다.
    :param data: 데이터 리스트
    :return:
    """
    review = []
    label = []
    for datum in data:

        review.append(datum[0])
        label.append(datum[1])
    # 각각 데이터, 레이블을 리턴
    return review, np.array(label)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()

        self.branch1x1 = nn.Conv2d(in_channels, 96, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(64, 96, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = nn.Conv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(96, 96, kernel_size=3, padding=1)

        self.branch7x7dbl_1 = nn.Conv2d(in_channels,32, kernel_size=1)
        self.branch7x7dbl_2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.branch7x7dbl_3 = nn.Conv2d(64,64,kernel_size=3,padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 96, kernel_size=1)

    def forward(self, x):

        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_Avg_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_Avg_pool = self.branch_pool(branch_Avg_pool)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)

        outputs = [branch1x1, branch5x5, branch3x3dbl,branch7x7dbl, branch_Avg_pool]
        return torch.cat(outputs, 1)

class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)

class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class Regression(nn.Module):
    """
    영화리뷰 예측을 위한 Regression 모델입니다.
    """
    def __init__(self, embedding_dim: int, max_length: int):
        """
        initializer
        :param embedding_dim: 데이터 임베딩의 크기입니다
        :param max_length: 인풋 벡터의 최대 길이입니다 (첫 번째 레이어의 노드 수에 연관)
        """
        super(Regression, self).__init__()
        self.embedding_dim = embedding_dim
        self.character_size = 251
        self.output_dim = 1  # Regression
        self.max_length = max_length

        self.incept1 = InceptionA(in_channels=384)
        self.incept2 = InceptionB(in_channels=256)
        self.incept3 = InceptionC(in_channels=384,channels_7x7=128)
        self.incept4 = InceptionD(in_channels=64)
        self.incept5 = InceptionE(in_channels=256)

        self.conv1 = nn.Conv2d(1, 384, kernel_size=5)
        self.conv2 = nn.Conv2d(448, 256, kernel_size=3,padding=3)
        self.conv3 = nn.Conv2d(736, 384, kernel_size=3,padding=2)
        self.conv4 = nn.Conv2d(768, 64, kernel_size=3,padding=4)
        self.conv5 = nn.Conv2d(640,256,kernel_size=3,padding=2)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1152, 1)

        # 임베딩
        self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)

    def forward(self, data: list):
        """
        :param data: 실제 입력값
        :return:
        """
        #print(data)
        # 임베딩의 차원 변환을 위해 배치 사이즈를 구합니다.
        batch_size = len(data)
        # list로 받은 데이터를 torch Variable로 변환합니다.

        data_in_torch = Variable(torch.from_numpy(np.array(data)).long())
        # 만약 gpu를 사용중이라면, 데이터를 gpu 메모리로 보냅니다.
        if GPU_NUM:
            data_in_torch = data_in_torch.cuda()
        # 뉴럴네트워크를 지나 결과를 출력합니다.

        embeds = self.embeddings(data_in_torch)

        #중요 코드
        embeds = embeds.view(batch_size,1,64,16)
        keep_prob = 0.5
        embeds = F.relu(self.mp((self.conv1(embeds))))
        embeds = (self.incept1(embeds))
        embeds = F.dropout(embeds, keep_prob)
        embeds = F.relu(self.mp((self.conv2(embeds))))
        embeds = (self.incept2(embeds))
        embeds = F.dropout(embeds, keep_prob)
        embeds = F.relu(self.mp((self.conv3(embeds))))
        embeds = (self.incept3(embeds))
        embeds = F.dropout(embeds, keep_prob)
        embeds = F.relu(self.mp((self.conv4(embeds))))
        embeds = (self.incept4(embeds))
        embeds = F.dropout(embeds, keep_prob)
        embeds = embeds.view(batch_size, -1)  # flatten the tensor
        embeds = self.fc(embeds)
        # 영화 리뷰가 1~10점이기 때문에, 스케일을 맞춰줍니다
        output2 = torch.sigmoid(embeds) * 9 + 1

        return output2


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=8)
    args.add_argument('--batch', type=int, default=16)
    args.add_argument('--strmaxlen', type=int, default=64)
    args.add_argument('--embedding', type=int, default=16)
    config = args.parse_args()

    softmax = torch.nn.Softmax()
    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/movie_review/'

    model = Regression(config.embedding, config.strmaxlen)
    if GPU_NUM:
        model = model.cuda()

    # DONOTCHANGE: Reserved for nsml use
    bind_model(model, config)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # DONOTCHANGE: They are reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())


    # 학습 모드일 때 사용합니다. (기본값)
    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = MovieReviewDataset(DATASET_PATH, config.strmaxlen)

        train_loader = DataLoader(dataset=dataset,
                                  batch_size=config.batch,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=2)
        total_batch = len(train_loader)

        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            for i, (data, labels) in enumerate(train_loader):

                predictions = model(data)

                label_vars = Variable(torch.from_numpy(labels))

                if GPU_NUM:
                    label_vars = label_vars.cuda()
                loss = criterion(predictions, label_vars)
                if GPU_NUM:
                    loss = loss.cuda()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print('Batch : ', i + 1, '/', total_batch,', MSE in this minibatch: ', loss.data[0])
                avg_loss += loss.data[0]
            print('epoch:', epoch, ' train_loss:', float(avg_loss/total_batch))
            # nsml ps, 혹은 웹 상의 텐서보드에 나타나는 값을 리포트하는 함수입니다.
            #
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss/total_batch), step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.0, 9.045), (0.0, 5.91), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            reviews = f.readlines()
        res = nsml.infer(reviews)
        print(res)