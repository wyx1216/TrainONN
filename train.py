import mlflow
import torch
import torchvision.datasets
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from mlflow import MlflowClient
import lightning as L
from torch.nn.functional import one_hot
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
from sklearn.datasets import make_classification
from torchvision.datasets import MNIST

from model import DNN, mzi_mesh, SimMZIMitrix, PPOFixedMZI, SimpleModel, TiledMZIMatrix


def get_simple_data(n_samples=1000,n_features=8):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,  # 总维度：8个特征
        n_informative=8,  # 有效特征：其中5个维度包含对分类有用的信息
        n_redundant=0,  # 冗余特征：其中2个是由有效特征线性组合产生的（模拟多重共线性）
        n_repeated=0,  # 重复特征：0个
        n_classes=2,  # 类别数量：2个分类 (0 和 1)
        weights=None,  # 类别均衡：正负样本各占 50%
        random_state=42  # 随机种子，确保每次生成的数据一致，方便复现和调试
    )
    X, y = torch.tensor(X, dtype=torch.float32).cuda(), torch.tensor(y, dtype=torch.long).cuda()
    # y = one_hot(y, num_classes=8)
    # y = torch.nn.functional.pad(y, (0, 6), "constant", 0)

    return X, y


if __name__ == '__main__':

    logger=MLFlowLogger(tracking_uri='sqlite:///D:/TrainONN-RL-w-o-NN/TrainONN-RL-w-o-NN/mlflow_new.db',
                        experiment_name = "PPOFixedMZI",
                        run_name="CE 直给; tiled; mnist; "
    )
    transform = torchvision.transforms.ToTensor()

    # 从官方训练集固定划分训练/验证集；官方测试集只在训练完成后评估一次。
    full_train_dataset = MNIST(root='FFA/data', download=True, train=True, transform=transform)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [55_000, 5_000],
        generator=torch.Generator().manual_seed(42),
    )
    test_dataset = MNIST(root='FFA/data', download=True, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    trainer = Trainer(
        max_epochs=20,
        logger=logger,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )
    #trainer=Trainer(max_epochs=20,logger=logger,check_val_every_n_epoch=1)
    # N = 40960
    # size = 8
    # X,y=get_simple_data(n_samples=N, n_features=size)
    # # torch.nn.functional.pad(y,(0,6))
    # train_dataset = TensorDataset(X,y)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    # X,y=get_simple_data(n_samples=1024, n_features=size)
    # val_dataset = TensorDataset(X,y)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

    # mzi = mzi_mesh(np.random.randn(10, 8), parallel=8)
    mzi = SimMZIMitrix(10, 8)
    tiled_mzi=TiledMZIMatrix(784,10,10,8).to("cuda:0")
    # 2. 动态获取正确的动作空间大小
    # in_blocks(98) * out_blocks(2) * 10 * 8 = 15680
    correct_act_space = tiled_mzi.in_blocks * tiled_mzi.out_blocks * tiled_mzi.layer_num * tiled_mzi.parallel
    ppo_agent = PPOFixedMZI(M_samples=16, act_space=correct_act_space ,  actor_lr=1e-2, mzi=tiled_mzi,
                    epsilon=0.2).cuda()
    simple_nn=SimpleModel()
    trainer.fit(ppo_agent, train_loader, val_loader)
    trainer.test(ppo_agent, dataloaders=test_loader)
    #& "D:\PycharmProjects\pythonProject\.venv\Scripts\python.exe" .\train.py
