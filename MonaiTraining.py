from __future__ import print_function, division
import torch
import os
import torch
import pandas as pd
import warnings
import torch.optim as optim
import matplotlib 
warnings.filterwarnings("ignore")
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import torch
import CustomDataLoaderCD as CDL
import monai as monai
import timeit
import logging
import numpy as np
from sklearn.metrics import accuracy_score
from torchsummary import summary
from pathlib import Path
import kornia.losses.focal as focal
from sklearn.metrics import roc_auc_score


def roc_var_calc(pred, label):
    """

    :param pred: list of predicticted values
    :param label: list of target values
    :return:
    """
    True_pos = 0
    False_pos = 0
    True_neg = 0
    False_neg = 0
    for i in range(len(label)):
        if label[i] == 1:
            if pred[i] == 1:
                True_pos += 1
            elif pred[i] == 0:
                False_neg += 1
        elif label[i] == 0:
            if pred[i] == 1:
                False_pos += 1
            elif pred[i] == 0:
                True_neg += 1
    print(True_pos, False_pos, True_neg, False_neg)
    return True_pos, False_pos, True_neg, False_neg


def roc_auc(pred, label):
    """

    :param pred: list of predicticted values
    :param label: list of target values
    :return:
    """
    True_pos, False_pos, True_neg, False_neg = roc_var_calc(pred, label)
    if True_pos == 0 and False_neg == 0:
        TPR = 0
    else:
        TPR = True_pos / (True_pos + False_neg)
    if True_neg == 0 and False_pos == 0:
        specificity = 0
    else:
        specificity = True_neg / (True_neg + False_pos)
    FPR = 1 - specificity
    print("TPR", TPR, "FPR", FPR, "specificity ", specificity)
    return TPR, FPR, specificity


def replace_batch_to(net, threeD):
    """

    :param net:
    :param threeD:
    :return:
    """
    if threeD == True:
        for child_name, child in net.named_children():
            if isinstance(child, torch.nn.BatchNorm3d):
                setattr(net, child_name, torch.nn.InstanceNorm3d(child.num_features, track_running_stats=False))
            else:
                replace_batch_to(child, threeD)

    else:
        for child_name, child in net.named_children():
            if isinstance(child, torch.nn.BatchNorm2d):
                setattr(net, child_name, torch.nn.InstanceNorm2d(child.num_features, track_running_stats=False))
            else:
                replace_batch_to(child, threeD)


class Model:
    """
    DenseNet model for binary classification 3D images

    setting:growth_rate=16, init_features=64
    loss: BCEWithLogitsLoss
    optim: Adam
    """

    def __init__(self, num_epochs, batch_size, lr, csv_train, csv_test, threeD, load, num_workers=3):
        """

        :param num_epochs:
        :param batch_size:
        :param lr:
        :param csv_train: csv with images and label
        :param csv_test: csv with images and label
        :param num_workers:
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers
        self.csv_train = csv_train
        self.csv_test = csv_test
        self.t_loss = []
        self.t_acc = []
        self.v_acc = []
        self.v_loss = []
        self.best_val_loss = 100
        self.threeD = threeD
        self.load = load

        self.column_name = ["Training True pos rate", 'Training False Positive rate',
                            'Training Specificify', "Validation True pos rate", 'Validation False Positive rate',
                            'Validation Specificify']

        self.result = pd.DataFrame(columns=self.column_name)

        self.data = CDL.Bms(csv_train,
                            transform=transforms.Compose([CDL.NormPad(), CDL.ToTensor()]))
        self.val = CDL.Bms(csv_test,
                           transform=transforms.Compose([CDL.NormPad(), CDL.ToTensor()]))

        self.dataloader = DataLoader(self.data, batch_size=self.batch_size,
                                     shuffle=True, num_workers=8)
        self.val_loader = DataLoader(self.val, batch_size=self.batch_size,
                                     shuffle=False, num_workers=8)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.threeD:
            self.net = monai.networks.nets.DenseNet(spatial_dims=3, in_channels=1, out_channels=1, growth_rate=32,
                                                    init_features=128, dropout_prob=0.5,
                                                    block_config=(6, 12, 24, 16)).to(self.device)
            replace_batch_to(self.net, threeD)
            self.net.to(self.device)
        else:
            print('else')
            self.net = monai.networks.nets.DenseNet(spatial_dims=2, in_channels=3, out_channels=1, growth_rate=32,
                                                    init_features=128, block_config=(6, 12, 48, 32),  dropout_prob=0.5
                                                    ).to(self.device)
            replace_batch_to(self.net, threeD)
            self.net.to(self.device)

        # self.pos_weight = torch.Tensor([10]).to(self.device)

        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.optimizer = torch.optim.AdamW(self.net.parameters(), self.lr)

        if self.load:
            ckp_path = "C:/Users/alexn/Desktop/best_9.pth"
            checkpoint = torch.load(ckp_path)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Training set dimension: ', self.data.__len__())
        print('Validation set dimension: ', self.val.__len__())

    def summary(self):
        summary(self.net, (3, 224, 224))

    def update_r(self, list_r):

        p = pd.DataFrame([list_r], columns=self.column_name)

        self.result = self.result.append(p)

        self.result = self.result.round(5)

        self.result.to_csv("C:/Users/alexn/Desktop/PetImages/Result.csv")

    def start_train(self):
        """
        Start the training
        :return:
        """
        for epoch in range(self.num_epochs):
            print(epoch)
            running_loss, y_true, y_pred, T_TPR, T_FPR, T_specificity = self.train()
            va_loss, v_true, v_pred, V_TPR, V_FPR, V_specificity = self.evaluate()
            res_list = [T_TPR, T_FPR, T_specificity, V_TPR, V_FPR, V_specificity]

            self.update_r(res_list)

            if V_TPR > 0.8 and V_specificity > 0.8:
                print('min')
                self.save_model_val(epoch)

            self.t_acc.append(accuracy_score(y_true, y_pred))
            self.t_loss.append(running_loss / self.data.__len__())
            self.v_acc.append(accuracy_score(v_true, v_pred))
            self.v_loss.append(va_loss / self.val.__len__())

            if epoch % 10 == 0:
                self.save_model(epoch)

            print("Accuracy on training set is", accuracy_score(y_true, y_pred), 'Loss: ',
                  running_loss / self.data.__len__(),
                  "Accuracy on training set is", accuracy_score(v_true, v_pred), 'Loss: ', va_loss / self.val.__len__())

            torch.save(self.net.state_dict(), "C:/Users/alexn/Desktop/PetImages/" + str(epoch) + ".pth")

        print('training loss:', self.t_loss)
        print('training acc:', self.t_acc)
        print('val loss:', self.v_loss)
        print('val acc:', self.v_acc)

    def train(self):
        """
        training function
        :return:
        """
        running_loss = 0
        y_true = []
        y_pred = []
        self.net.train()

        for i_batch, sample_batched in enumerate(self.dataloader):
            inputs, labels = sample_batched['image'].to(self.device), sample_batched['label'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            labels = labels.to(torch.float32)
            labels = labels.view(-1, 1)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            pred = np.round(torch.sigmoid(outputs.detach().cpu()).tolist())
            target = labels.tolist()
            y_true.extend(target[0])
            y_pred.extend(pred[0])
        print('roc_auc_score', roc_auc_score(y_true, y_pred))
        TPR, FPR, specificity = roc_auc(y_pred, y_true)
        return running_loss, y_true, y_pred, TPR, FPR, specificity

    def evaluate(self):
        """
        Validation function
        :return:
        """
        self.net.eval()
        va_loss = 0
        v_true = []
        v_pred = []
        for i_batch, sample_batched in enumerate(self.val_loader):
            inputs, labels = sample_batched['image'].to(self.device), sample_batched['label'].to(self.device)
            with torch.no_grad():
                labels = labels.to(torch.float32)
                labels = labels.view(-1, 1)
                outputs = self.net(inputs)
                va_loss += self.criterion(outputs, labels).item()
                pred = np.round(torch.sigmoid(outputs.detach().cpu()).tolist())
                target = labels.tolist()
                v_true.extend(target[0])
                v_pred.extend(pred[0])
        print('roc_auc_score', roc_auc_score(v_true, v_pred))
        TPR, FPR, specificity = roc_auc(v_pred, v_true)

        return va_loss, v_true, v_pred, TPR, FPR, specificity

    def save_model(self, epoch):
        """
        Save the actual state of the network
        :param epoch:
        :return:
        """
        PATH = 'C:/Users/alexn/Desktop/PetImages/' + str(10 * epoch) + '.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)

    def save_model_val(self, epoch):
        """
        Save the actual state of the network
        :param epoch:
        :return:
        """
        PATH = 'C:/Users/alexn/Desktop/PetImages/' + 'best_' + str(epoch) + '.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),

        }, PATH)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", default=100, help="Set number of epochs", type=int)
    parser.add_argument("batch_size", default=1, help="Set the batch size", type=int)
    parser.add_argument("learning_rate", default=1e-5, help="Set the learning rate", type=float)
    parser.add_argument("train_set", help="Path to dataset on local disk", type=Path)
    parser.add_argument("val_set", help="Path to dataset on local disk", type=Path)
    parser.add_argument("--threeD", default=False, help="Set number of epochs", type=bool, required=False)
    parser.add_argument("--load", default=False, help="Set if you want to load weight", type=bool, required=False)

    args = parser.parse_args()
    test = Model(args.epochs, args.batch_size, args.learning_rate, args.train_set, args.val_set, args.threeD, args.load)
    test.start_train()

