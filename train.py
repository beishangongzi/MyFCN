import os.path

import PIL.Image
import numpy as np
import torch.cuda
from torch import nn, optim
from torch.utils.data import DataLoader

import dataset
import utils
from model.FCN import FCN32, FCN16, FCN8


class Train:
    def __init__(self, dataset_path, model, batch_size, shuffle, mode="train"):

        self.dataset = dataset.ObtTrainDataset(dataset_path, mode=mode)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using {self.device}")
        self.model = model.to(self.device)

    def train(self, save_name):
        epoch = 100
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        dl = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True)
        for i in range(epoch):
            print("------------{} begin--------------".format(i))
            self.model.train()
            running_loss = 0.0
            j = 0
            for data in dl:
                j += 1
                inputs, target = data
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                target = torch.squeeze(target, 1).long().to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.cpu().item()
            print(running_loss / j / self.batch_size)
            if (i + 1) % 10 == 0:
                torch.save(self.model.state_dict(), f"models/{save_name}_{i}.pth")
        torch.save(self.model.state_dict(), f"models/{save_name}_last_.pth")

    def test(self, save_name):
        """
        this is shit. don't see it.
        :param test_path:
        :param model:
        :return:
        """
        self.model.load_state_dict(torch.load(save_name))
        dl = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            for data in dl:
                inputs, names = data
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                batch = outputs.size()[0]
                for i in range(batch):
                    output = outputs[i]
                    output = output.cpu().numpy()
                    output = np.argmax(output, 0)
                    output = utils.Utils.to_color(output)
                    pred_name = os.path.join("data/obt/testImagesPreds", names[i] + ".png")
                    PIL.Image.fromarray(output).save(pred_name)
                    ground_truth = np.load("data/obt/testImagesMasks/" + names[i]).squeeze()
                    ground_truth = utils.Utils.to_color(ground_truth)
                    truth_name = os.path.join("data/obt/testImagesMasks", names[i] + ".png")
                    PIL.Image.fromarray(ground_truth).save(truth_name)


def run(model_name, save_name, mode, load_name=None):
    dataset = "data/obt/image"
    models = {"FCN32": FCN32, "FCN16": FCN16, "FCN8": FCN8}

    model = models.get(model_name)(256, 5)
    train = Train(dataset, model, 8, True, mode=mode)
    if mode == "train":
        train.train(save_name)
    else:
        train.test(load_name)
