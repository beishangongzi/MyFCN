import PIL.Image
import numpy as np
import torch.cuda
from torch import nn, optim
from torch.utils.data import DataLoader

import dataset
import utils
from model.FCN import FCN32


class Train:
    def __init__(self, dataset_path, model, batch_size, shuffle):

        self.dataset = dataset.ObtTrainDataset(dataset_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using {self.device}")
        self.model = model.to(self.device)

    def train(self):
        epoch = 100
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        dl = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False)
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
            if (i+1) % 10 == 0:
                torch.save(self.model.state_dict(), f"/models/obt_10_{i}.pth")
        torch.save(self.model.state_dict(), f"models/obt_10_last_.pth")

    def test(self, test_path, model):
        """
        this is shit. don't see it.
        :param test_path:
        :param model:
        :return:
        """
        dl = DataLoader(dataset.ObtTrainDataset(test_path, mode="test"), batch_size=10, shuffle=False)
        self.model.load_state_dict(torch.load("obt_10_9.pth"))

        for data in dl:
            inputs, target = data
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
            print(outputs.size())
            # print(outputs[0].size())
            n = outputs[0].detach().numpy()
            a = np.argmax(n, 0)
            a = utils.Utils.to_color(a)
            PIL.Image.fromarray(a).save("r.png")
            print(inputs[0].size())
            a = np.load("data/obt/testMasks/" + target[0])
            print(a.shape)
            a = utils.Utils.to_color(a.squeeze())
            PIL.Image.fromarray(a).save("rr.png")
            break




def test():
    dataset = "data/obt/image"
    model = FCN32(256, 5)
    train = Train(dataset, model, 8, True)
    train.train()


if __name__ == '__main__':
    test()