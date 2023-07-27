import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
import wandb
from dotenv import load_dotenv
from loguru import logger
from src.data.dataset import CustomDataset
from src.utils import SIMCLR_AUGMENTATIONS, load_config
from src.model.model import Vision
from src.loss import SimCLRLoss
from config import Cfg


load_dotenv()
wandb.init(
    project="simclr",
    name="experiment-1",
    mode="online"
)
wandb.config.update(load_config("config.yaml"))
config: Cfg = wandb.config
logger.add("logs/{time}.log")

dataset = CustomDataset(root="src/data/images", transforms=SIMCLR_AUGMENTATIONS)
dataloader = DataLoader(dataset=dataset, batch_size=config.batch_size)
vision = Vision()
loss_fn = SimCLRLoss(batch_size=config.batch_size, temperature=config.temperature)
metric = MeanMetric()

for epoch in range(config.epochs):
    for image1, image2 in dataloader:
        representation1, representation2 = vision(image1), vision(image2)
        loss = loss_fn(representation1, representation2)
        loss.backward()
        with torch.no_grad():
            assert type(vision.weight.grad) is torch.Tensor
            grad_norm = vision.weight.grad.norm()
            wandb.log(
                data={
                    "weight": {
                        "norm": vision.weight.norm(),
                        "topology": wandb.Histogram(vision.weight),
                    },
                    "gradient": {
                        "norm": vision.weight.grad.norm(),
                        "topology": wandb.Histogram(vision.weight.grad),
                    }
                },
                step=epoch,
                commit=False if (epoch+1) % config.log_interval == 0 else True
            )
            vision.weight -= config.lr * vision.weight.grad
            vision.weight.grad.zero_()
            metric.update(loss)
            if (epoch+1) % config.log_interval == 0:
                wandb.log(
                    data={
                        "loss": metric.compute() 
                    },
                    step=epoch,
                    commit=True
                )
                metric.reset()
            
        logger.info(f"epoch: {epoch}\tloss: {loss:.4f}\tgrad_norm: {grad_norm:.4f}")
