import os.path

import PIL.Image
import numpy as np
import torch.cuda
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from torchvision import transforms

import dataset
import utils
from model.FCN import FCN32, FCN16, FCN8
from model.Unet.Unet import Unet


class Train:
    def __init__(self, dataset_path, model, batch_size, shuffle, mode="train"):

        self.dataset = dataset.ObtTrainDataset(dataset_path, mode=mode)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using {self.device}")
        self.model = model.to(self.device)

    def train(self, save_name, save_freq, lr, epoch):
        epoch = epoch
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
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
            if (i + 1) % save_freq == 0:
                torch.save(self.model.state_dict(), f"models/{save_name}.pth")
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
                    target = torch.tensor(np.load("data/obt/testImageMasks/" + names[i]).squeeze())
                    confmat = ConfusionMatrix(num_classes=5)

                    output = output.cpu().numpy()
                    output = np.argmax(output, 0)
                    morphology_x = output
                    print(confmat(torch.tensor(output).reshape([1, -1]), target.reshape([1, -1])))
                    # print(confmat(target.reshape([1, -1]), target.reshape([1, -1])))
                    print("\n\n\n----------------")
                    output = utils.Utils.to_color(output)

                    pred_name = os.path.join("data/obt/testImagePreds", names[i] + ".png")
                    PIL.Image.fromarray(output).save(pred_name)
                    ground_truth = np.load("data/obt/testImageMasks/" + names[i]).squeeze()
                    ground_truth = utils.Utils.to_color(ground_truth)
                    truth_name = os.path.join("data/obt/testImageMasks", names[i] + ".png")
                    PIL.Image.fromarray(ground_truth).save(truth_name)
                    morphology_close = utils.Morphology.close(morphology_x.astype("uint8"))
                    morphology_close = utils.Utils.to_color(morphology_close)
                    morphology_close_name = os.path.join("data/obt/testImageMorphology/close", names[i] + ".png")
                    PIL.Image.fromarray(morphology_close).save(morphology_close_name)

                    morphology_open = utils.Morphology.open(morphology_x.astype("uint8"))
                    morphology_open = utils.Utils.to_color(morphology_open)
                    morphology_open_name = os.path.join("data/obt/testImageMorphology/open", names[i] + ".png")
                    PIL.Image.fromarray(morphology_open).save(morphology_open_name)

                    morphology_erode = utils.Morphology.erode(morphology_x.astype("uint8"))
                    morphology_erode = utils.Utils.to_color(morphology_erode)
                    morphology_erode_name = os.path.join("data/obt/testImageMorphology/erode", names[i] + ".png")
                    PIL.Image.fromarray(morphology_erode).save(morphology_erode_name)

                    morphology_dilate = utils.Morphology.dilate(morphology_x.astype("uint8"))
                    morphology_dilate = utils.Utils.to_color(morphology_dilate)
                    morphology_dilate_name = os.path.join("data/obt/testImageMorphology/dilate", names[i] + ".png")
                    PIL.Image.fromarray(morphology_dilate).save(morphology_dilate_name)


def run(model_name, save_name, mode, dataset, backbone, lr, epoch, load_name=None, save_freq=20):
    models = {"FCN32": FCN32, "FCN16": FCN16, "FCN8": FCN8, "Unet": Unet}

    model = models.get(model_name)(5, backbone)
    train = Train(dataset, model, 8, True, mode=mode)
    if mode == "train":
        train.train(save_name, save_freq,
                    lr=lr,
                    epoch=epoch)
    else:
        train.test(load_name)
