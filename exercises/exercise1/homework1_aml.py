import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn
from torchvision import transforms
from torchvision.models import alexnet
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import VisionDataset
from PIL import Image
import os
import os.path
from sklearn.model_selection import train_test_split
import numpy as np
import random

##################### ARGS #####################
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
NUM_CLASSES = 102  # 101 + 1: There is am extra Background class that should be removed

BATCH_SIZE = 256  # Higher batch sizes allows for larger learning rates. An empirical heuristic suggests that, when changing
# the batch size, learning rate should change by the same factor to have comparable results

LR = 1e-3  # The initial Learning Rate
MOMENTUM = 0.9  # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default

NUM_EPOCHS = 30  # Total number of training epochs (iterations over dataset)
STEP_SIZE = (
    20  # How many epochs before decreasing learning rate (if using a step-down policy)
)
GAMMA = 0.1  # Multiplicative factor for learning rate step-down

LOG_FREQUENCY = 10
WORKERS = 4  # Number of workers on the data loader (4*num_GPUs)
DATA_DIR = "Caltech101"


##################### DATA #####################
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class Caltech(VisionDataset):
    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        augmentation=False,
    ):
        super(Caltech, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.split = split  # This defines the split you are going to use
        # (split files are called 'train.txt' and 'test.txt')

        self.augmentation = augmentation

        self.images = []
        self.labels = []
        self.classes = {}

        # Read split files and assign labels
        split_file = os.path.join(root, f"{split}.txt")
        with open(split_file, "r") as f:
            for line in f:
                image_path = line.strip()
                class_name = os.path.basename(os.path.dirname(image_path))
                if class_name not in self.classes:
                    self.classes[class_name] = len(self.classes)
                self.images.append(image_path)
                self.labels.append(self.classes[class_name])

    def __getitem__(self, index):
        """
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        image = pil_loader(
            os.path.join(self.root, "101_ObjectCategories", self.images[index])
        )
        label = self.labels[index]

        # Use data augmentation for training data
        if self.split == "train" and self.augmentation:
            if random.random() > 0.25:
                # Randomly select an augmentation technique
                augmentation_method = random.choice(["flip", "rotate", "color_jitter"])

                if augmentation_method == "flip":
                    # Random horizontal flip
                    if random.random() > 0.5:
                        image = transforms.functional.hflip(image)
                    else:
                        image = transforms.functional.vflip(image)

                elif augmentation_method == "rotate":
                    # Random rotation
                    degrees = random.choice([90, 180, 270])
                    image = transforms.functional.rotate(image, degrees)

                elif augmentation_method == "color_jitter":
                    # Random color jitter
                    brightness = random.uniform(0.8, 1.2)
                    contrast = random.uniform(0.8, 1.2)
                    saturation = random.uniform(0.8, 1.2)
                    hue = random.uniform(-0.1, 0.1)
                    image = transforms.functional.adjust_brightness(image, brightness)
                    image = transforms.functional.adjust_contrast(image, contrast)
                    image = transforms.functional.adjust_saturation(image, saturation)
                    image = transforms.functional.adjust_hue(image, hue)

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        """
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        """
        return len(self.images)


def load_data(
    mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), validation=True, augmentation=False
):
    # Define transforms for training phase
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),  # Resizes short size of the PIL image to 256
            transforms.CenterCrop(224),  # Crops a central square patch of the image
            # 224 because torchvision's AlexNet needs a 224x224 input!
            # Remember this when applying different transformations, otherwise you get an error
            transforms.ToTensor(),  # Turn PIL Image to torch.Tensor
            transforms.Normalize(
                mean, std
            ),  # Normalizes tensor with mean and standard deviation
        ]
    )
    # Define transforms for the evaluation phase
    eval_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Prepare Pytorch train/test Datasets
    train_dataset = Caltech(
        DATA_DIR, split="train", transform=train_transform, augmentation=augmentation
    )
    test_dataset = Caltech(DATA_DIR, split="test", transform=eval_transform)

    if validation:
        train_indexes, val_indexes = train_test_split(
            np.arange(len(train_dataset)),
            stratify=train_dataset.labels,
            test_size=0.25,
            random_state=42,
        )

        new_train_dataset = Subset(train_dataset, train_indexes)
        val_dataset = Subset(train_dataset, val_indexes)

        train_size = len(new_train_dataset)
        test_size = len(test_dataset)
        val_size = len(val_dataset)

        # Check dataset sizes
        print("Train Dataset: {}".format(train_size))
        print("Valid Dataset: {}".format(val_size))
        print("Test Dataset: {}".format(test_size))

        # Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
        train_dataloader = DataLoader(
            new_train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=WORKERS,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=WORKERS,
        )
        return (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            train_size,
            val_size,
            test_size,
        )
    else:
        train_size = len(train_dataset)
        test_size = len(test_dataset)
        # Check dataset sizes
        print("Train Dataset: {}".format(train_size))
        print("Test Dataset: {}".format(test_size))

        # Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=WORKERS,
            drop_last=True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=WORKERS,
        )

        return train_dataloader, test_dataloader, train_size, test_size


##################### TRAINING #####################


def train(
    model,
    loss_fn,
    optimizer,
    scheduler,
    train_dataloader,
    train_size,
    val_dataloader=None,
    val_size=None,
    best_model_path="best_model.pth",
):
    start = time.time()  # Start time
    model = model.to(DEVICE)  # Move model to device
    max_val_acc = float("-inf")
    cudnn.benchmark = True  # Calling this optimizes runtime

    print("-" * 10)
    # Start iterating over the epochs
    for epoch in range(NUM_EPOCHS):
        print(
            "Epoch {}/{}, LR = {}".format(
                epoch + 1, NUM_EPOCHS, scheduler.get_last_lr()
            )
        )
        running_loss = 0.0

        # TRAIN
        model.train()  # Sets module in training mode
        for images, labels in train_dataloader:
            # Bring data over the device of choice
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # PyTorch, by default, accumulates gradients after each backward pass
            # We need to manually set the gradients to zero before starting a new iteration
            optimizer.zero_grad()  # Zero-ing the gradients

            # Forward pass to the network
            outputs = model(images)

            # Compute loss based on output and ground truth
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()

            # Compute gradients for each layer and update weights
            loss.backward()  # backward pass: computes gradients
            optimizer.step()  # update weights based on accumulated gradients

        print("Epoch Loss:", running_loss / train_size)

        # VALIDATE
        if val_dataloader is not None:
            model.eval()  # Set Network to evaluation mode
            running_corrects = 0
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_dataloader:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)

                    # Forward Pass
                    outputs = model(images)

                    # Get predictions
                    _, preds = torch.max(outputs, 1)

                    # Update Corrects
                    running_corrects += torch.sum(preds == labels.data).item()
                    val_loss += loss_fn(outputs, labels).item()

            # Calculate Accuracy
            val_acc = running_corrects / float(val_size)
            val_loss /= val_size
            print("Val Accuracy: {}".format(val_acc))
            print("Val Loss: {}".format(val_loss))

            if val_acc > max_val_acc:
                max_val_acc = val_acc
                print("New Best Val Acc!")
                torch.save(model.state_dict(), best_model_path)

            print("Best Val Acc: {}".format(max_val_acc))
        print("-" * 10)

        # Step the scheduler
        scheduler.step()

    time_elapsed = time.time() - start
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Acc: {max_val_acc}")

    # load best model weights
    model.load_state_dict(torch.load(best_model_path))
    return model


def validation(model, val_dataloader, val_size):
    model = model.to(DEVICE)  # this will bring the network to GPU if DEVICE is cuda
    model.train(False)  # Set Network to evaluation mode

    running_corrects = 0
    for images, labels in val_dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward Pass
        outputs = model(images)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # Update Corrects
        running_corrects += torch.sum(preds == labels.data).data.item()

    # Calculate Accuracy
    accuracy = running_corrects / float(val_size)

    print("Val Accuracy: {}".format(accuracy))
    return accuracy


def test(model, test_dataloader, test_size):
    model = model.to(DEVICE)  # this will bring the network to GPU if DEVICE is cuda
    model.train(False)  # Set Network to evaluation mode

    running_corrects = 0
    for images, labels in tqdm(test_dataloader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward Pass
        outputs = model(images)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # Update Corrects
        running_corrects += torch.sum(preds == labels.data).data.item()

    # Calculate Accuracy
    accuracy = running_corrects / float(test_size)

    print("Test Accuracy: {}".format(accuracy))


##################### MODEL #####################


def scratch():

    net = alexnet()  # Loading AlexNet model

    # AlexNet has 1000 output neurons, corresponding to the 1000 ImageNet's classes
    # We need 101 outputs for Caltech-101
    net.classifier[6] = nn.Linear(
        4096, NUM_CLASSES
    )  # nn.Linear in pytorch is a fully connected layer
    # The convolutional layer is nn.Conv2d

    # We just changed the last layer of AlexNet with a new fully connected layer with 101 outputs
    # It is strongly suggested to study torchvision.models.alexnet source code

    # Define loss function
    criterion = nn.CrossEntropyLoss()  # for classification, we use Cross Entropy

    # Choose parameters to optimize
    # To access a different set of parameters, you have to access submodules of AlexNet
    # (nn.Module objects, like AlexNet, implement the Composite Pattern)
    # e.g.: parameters of the fully connected layers: net.classifier.parameters()
    # e.g.: parameters of the convolutional layers: look at alexnet's source code ;)
    parameters_to_optimize = (
        net.parameters()
    )  # In this case we optimize over all the parameters of AlexNet

    # Define optimizer
    # An optimizer updates the weights based on loss
    # We use SGD with momentum
    optimizer = optim.SGD(
        parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    # Define scheduler
    # A scheduler dynamically changes learning rate
    # The most common schedule is the step(-down), which multiplies learning rate by gamma every STEP_SIZE epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        train_size,
        val_size,
        test_size,
    ) = load_data(validation=True)
    net = train(
        net,
        criterion,
        optimizer,
        scheduler,
        train_dataloader=train_dataloader,
        train_size=train_size,
        val_dataloader=val_dataloader,
        val_size=val_size,
    )
    test(net, test_dataloader, test_size)


def finetuning(augmentation=False):

    net_ft = alexnet(weights="IMAGENET1K_V1")  # Loading AlexNet model

    # AlexNet has 1000 output neurons, corresponding to the 1000 ImageNet's classes
    # We need 101 outputs for Caltech-101
    net_ft.classifier[6] = nn.Linear(
        4096, NUM_CLASSES
    )  # nn.Linear in pytorch is a fully connected layer
    # The convolutional layer is nn.Conv2d

    # We just changed the last layer of AlexNet with a new fully connected layer with 101 outputs
    # It is strongly suggested to study torchvision.models.alexnet source code

    # Define loss function
    criterion_ft = nn.CrossEntropyLoss()  # for classification, we use Cross Entropy

    # Choose parameters to optimize
    # To access a different set of parameters, you have to access submodules of AlexNet
    # (nn.Module objects, like AlexNet, implement the Composite Pattern)
    # e.g.: parameters of the fully connected layers: net.classifier.parameters()
    # e.g.: parameters of the convolutional layers: look at alexnet's source code ;)
    parameters_to_optimize_ft = (
        net_ft.parameters()
    )  # In this case we optimize over all the parameters of AlexNet

    # Define optimizer
    # An optimizer updates the weights based on loss
    # We use SGD with momentum
    optimizer_ft = optim.SGD(
        parameters_to_optimize_ft, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )

    # Define scheduler
    # A scheduler dynamically changes learning rate
    # The most common schedule is the step(-down), which multiplies learning rate by gamma every STEP_SIZE epochs
    scheduler_ft = optim.lr_scheduler.StepLR(
        optimizer_ft, step_size=STEP_SIZE, gamma=GAMMA
    )

    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        train_size,
        val_size,
        test_size,
    ) = load_data(
        validation=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        augmentation=augmentation,
    )
    net_ft = train(
        net_ft,
        criterion_ft,
        optimizer_ft,
        scheduler_ft,
        train_dataloader=train_dataloader,
        train_size=train_size,
        val_dataloader=val_dataloader,
        val_size=val_size,
        best_model_path="best_model_ft.pth",
    )
    test(net_ft, test_dataloader, test_size)


##################### MAIN #####################

if __name__ == "__main__":
    # scratch()
    finetuning(augmentation=True)
