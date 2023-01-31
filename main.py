import torch.nn
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from itertools import chain
import os
from matplotlib import pyplot as plt


class LocalSTG(torch.nn.Module):
    def __init__(self):
        super(LocalSTG, self).__init__()
        self._sqrt_2 = math.sqrt(2)
        self._sigma = .5
        self.mu_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.Tanh(),
            nn.Linear(128, 784),
            nn.Tanh()
        )
        self.mu_net.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.01)
            if 'bias' in vars(m).keys():
                m.bias.data.fill_(0.0)

    def forward(self, x):
        noise = torch.normal(mean=0, std=self._sigma, size=x.size(), generator=torch.Generator(device=x.device).manual_seed(0), device=x.device)
        mu = self.mu_net(x)
        z = mu + self._sigma * noise * self.training
        sparse_x = x * self.hard_sigmoid(z)
        return mu, sparse_x

    def hard_sigmoid(self, x):
        return torch.clamp(x + self._sigma, 0.0, 1.0)

    def regularization(self, mu, reduction_func=torch.mean):
        return reduction_func(0.5 - 0.5 * torch.erf((-1 / 2 - mu) / (self._sigma * self._sqrt_2)))

    def gates(self, x):
        with torch.no_grad():
            gates = self.hard_sigmoid(self.mu_net(x))
        return gates


class MNISTTab(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y

    def __getitem__(self, index: int):
        return torch.tensor(self.X[index]).float(), torch.tensor(self.Y[index]).long()

    def __len__(self):
        return len(self.X)

    @classmethod
    def setup(cls, data_dir, train_subset=10000, test_subset=1000):
        X_train = MNIST(data_dir, train=True).data.reshape(-1, 784).cpu().numpy()
        X_test = MNIST(data_dir, train=False).data.reshape(-1, 784).cpu().numpy()
        Y_train = MNIST(data_dir, train=True).targets.cpu().numpy()
        Y_test = MNIST(data_dir, train=False).targets.cpu().numpy()
        X_train = X_train[:train_subset]
        Y_train = Y_train[:train_subset]

        X_test = X_test[:test_subset]
        Y_test = Y_test[:test_subset]

        X_train = X_train / 255.
        X_test = X_test / 255.

        return (cls(X_train, Y_train), cls(X_test, Y_test))


class MNISTExample(LightningModule):
    def __init__(self):
        super().__init__()
        self.train_dataset, self.test_dataset = MNISTTab.setup("C:/data/fs/mnist")

        # model:
        self.local_stg = LocalSTG()
        self.classifier = torch.nn.Linear(784, 10)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1024, drop_last=True, shuffle=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1000, drop_last=False, shuffle=False, num_workers=0)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # inference:
        mu, sparse_x = self.local_stg(x)
        logits = self.classifier(sparse_x)
        y_hat = torch.argmax(logits, dim=-1)
        # losses:
        ce_loss = F.cross_entropy(logits, y)
        regularization_loss = self.local_stg.regularization(mu)
        accuracy = torch.mean((y == y_hat).float())
        self.log("train/ce_loss", ce_loss.item())
        self.log("train/reg_loss", regularization_loss.item())
        self.log("train/accuracy", accuracy.item())
        return ce_loss + 5. * regularization_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, sparse_x = self.local_stg(x)
        logits = self.classifier(sparse_x)
        y_hat = torch.argmax(logits, dim=-1)
        ce_loss = F.cross_entropy(logits, y)
        accuracy = torch.mean((y == y_hat).float())
        self.log('val/accuracy', accuracy)
        self.log('val/ce_loss', ce_loss)

        gates = self.local_stg.gates(x)

        for i, (gates_i, y_i, x_i, gated_x_i, y_hat_i) in enumerate(zip(gates, y, x, sparse_x, y_hat)):
            y_i = y_i.cpu().numpy()
            gates_img = gates_i.reshape(28, 28).cpu().numpy()
            orig_img = x_i.reshape(28, 28).cpu().numpy()
            gated_img = gated_x_i.reshape(28, 28).cpu().numpy()
            self.save_image(gates_img, f'gates_sample_{i}_y_{y_i}_y_hat_{y_hat_i}_batch_idx_{batch_idx}.png')
            self.save_image(orig_img, f'orig_sample_{i}_y_{y_i}_y_hat_{y_hat_i}_batch_idx_{batch_idx}.png')
            self.save_image(gated_img, f'result_sample_{i}_y_{y_i}_y_hat_{y_hat_i}_batch_idx_{batch_idx}.png')
            if i == 20: break  # plot only a subset of images

        return ce_loss

    def configure_optimizers(self):
        return torch.optim.SGD(chain(self.local_stg.parameters(), self.classifier.parameters()), lr=.1)

    def save_image(self, image, name):
        os.makedirs("samples", exist_ok=True)
        plt.clf()
        fig, ax = plt.subplots()
        ax.set_title(name)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        plt.imshow(image, cmap='gray')
        plt.savefig(f"samples/{name}")
        plt.close()


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = MNISTExample()
    seed_everything(0)
    logger = TensorBoardLogger(save_dir='.', name="mnist", log_graph=False)
    trainer = Trainer(
        max_epochs=1000,
        deterministic=True,
        logger=True,
        log_every_n_steps=10,
        check_val_every_n_epoch=100,
        enable_checkpointing=False,
        callbacks=LearningRateMonitor(logging_interval='step'))
    trainer.logger = logger
    trainer.fit(model)
