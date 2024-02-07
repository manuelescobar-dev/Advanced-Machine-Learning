""" # Ignore depreciation warnings
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) """

import torch
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.backends import cudnn
from colorama import Fore
import torch
import torch.nn as nn
from torch.autograd import Variable
from src import resnetMod
from src.gtea_dataset import GTEA61
from src.spatial_transforms import (
    Compose,
    ToTensor,
    CenterCrop,
    Scale,
    Normalize,
    MultiScaleCornerCrop,
    RandomHorizontalFlip,
)

###### SET PARAMETERS ######

# use GPU if available
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Data
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data/GTEA61/")  # path dataset

# All this param can be change!
NUM_CLASSES = 61
BATCH_SIZE = 64
LR = 0.001  # The initial Learning Rate
MOMENTUM = 0.9  # Hyperparameter for SGD, keep this at 0.9 when using SGD
WEIGHT_DECAY = 4e-5  # Regularization, you can keep this at the default
NUM_EPOCHS = 200  # Total number of training epochs (iterations over dataset)
STEP_SIZE = [
    25,
    75,
    150,
]  # How many epochs before decreasing learning rate (if using a step-down policy)
GAMMA = 0.1  # Multiplicative factor for learning rate step-down
MEM_SIZE = 512  # Dim of internal state of LSTM or ConvLSTM
SEQ_LEN = 3  # Num Frames

# this dictionary is needed for the logger class
parameters = {
    "DEVICE": DEVICE,
    "NUM_CLASSES": NUM_CLASSES,
    "BATCH_SIZE": BATCH_SIZE,
    "LR": LR,
    "MOMENTUM": MOMENTUM,
    "WEIGHT_DECAY": WEIGHT_DECAY,
    "NUM_EPOCHS": NUM_EPOCHS,
    "STEP_SIZE": STEP_SIZE,
    "GAMMA": GAMMA,
    "MEM_SIZE": MEM_SIZE,
    "SEQ_LEN": SEQ_LEN,
}

###### Data ######


def load_data():
    # Normalize
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Training Spatial Transform (Random Crop)
    spatial_transform = Compose(
        [
            Scale(256),
            RandomHorizontalFlip(),
            MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224),
            ToTensor(),
            normalize,
        ]
    )

    # Validation Spatial Transform (Center Crop)
    spatial_transform_val = Compose(
        [Scale(256), CenterCrop(224), ToTensor(), normalize]
    )

    # Prepare Pytorch train/test Datasets
    train_dataset = GTEA61(
        DATA_DIR, split="train", transform=spatial_transform, seq_len=SEQ_LEN
    )
    test_dataset = GTEA61(
        DATA_DIR, split="test", transform=spatial_transform_val, seq_len=SEQ_LEN
    )

    # Check dataset sizes
    print("Train Dataset: {}".format(len(train_dataset)))
    print("Test Dataset: {}".format(len(test_dataset)))

    # Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    train_size = len(train_dataset)
    test_size = len(test_dataset)
    return train_loader, val_loader, train_size, test_size


###### Model ######


# LSTM
class MyLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(MyLSTMCell, self).__init__()

    def forward(self, x, state):
        if state is None:
            state = (
                Variable(torch.randn(x.size(0), x.size(1)).cuda()),
                Variable(torch.randn(x.size(0), x.size(1)).cuda()),
            )

        ##################################
        # You should implement this part #
        ##################################

        return None, None


# ConvLSTM
class MyConvLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size=3, stride=1, padding=1):
        super(MyConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv_i_xx = nn.Conv2d(
            input_size,
            hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv_i_hh = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.conv_f_xx = nn.Conv2d(
            input_size,
            hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv_f_hh = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.conv_c_xx = nn.Conv2d(
            input_size,
            hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv_c_hh = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        self.conv_o_xx = nn.Conv2d(
            input_size,
            hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv_o_hh = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )

        torch.nn.init.xavier_normal_(self.conv_i_xx.weight)
        torch.nn.init.constant_(self.conv_i_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_i_hh.weight)

        torch.nn.init.xavier_normal_(self.conv_f_xx.weight)
        torch.nn.init.constant_(self.conv_f_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_f_hh.weight)

        torch.nn.init.xavier_normal_(self.conv_c_xx.weight)
        torch.nn.init.constant_(self.conv_c_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_c_hh.weight)

        torch.nn.init.xavier_normal_(self.conv_o_xx.weight)
        torch.nn.init.constant_(self.conv_o_xx.bias, 0)
        torch.nn.init.xavier_normal_(self.conv_o_hh.weight)

    def forward(self, x, state):
        if state is None:
            state = (
                Variable(
                    torch.randn(x.size(0), x.size(1), x.size(2), x.size(3)).cuda()
                ),
                Variable(
                    torch.randn(x.size(0), x.size(1), x.size(2), x.size(3)).cuda()
                ),
            )

        ht_1, ct_1 = state
        it = torch.sigmoid(self.conv_i_xx(x) + self.conv_i_hh(ht_1))
        ft = torch.sigmoid(self.conv_f_xx(x) + self.conv_f_hh(ht_1))
        ct_tilde = torch.tanh(self.conv_c_xx(x) + self.conv_c_hh(ht_1))
        ct = (ct_tilde * it) + (ct_1 * ft)
        ot = torch.sigmoid(self.conv_o_xx(x) + self.conv_o_hh(ht_1))
        ht = ot * torch.tanh(ct)
        return ht, ct


# Network
class ourModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=512, homework_step=0, DEVICE=""):
        super(ourModel, self).__init__()
        self.DEVICE = DEVICE
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(
            True, True
        )  # Uses pretrained resnet34 (True) and shows progress (True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight  # Last layer weights
        self.homework_step = homework_step
        if self.homework_step == 1:
            self.lstm_cell = nn.LSTMCell(512, mem_size)
        elif self.homework_step == 2:
            self.lstm_cell = MyConvLSTMCell(512, mem_size)

        self.avgpool = nn.AvgPool2d(7)  # Average pooling layer 7x7
        self.dropout = nn.Dropout(0.7)  # Dropout layer
        self.fc = nn.Linear(mem_size, self.num_classes)  # Fully connected layer
        self.classifier = nn.Sequential(self.dropout, self.fc)  # Classifier

    def forward(self, inputVariable):
        """
        inputVariable: (Frames, BS, C, W, H)
        C: Channels
        BS: Batch Size
        Frames: Number of frames
        W: Width
        H: Height
        """
        # Learning without Temporal information (mean)
        if self.homework_step == 0:
            # Initialize video level features, which is the mean of the frames
            video_level_features = torch.zeros(
                (inputVariable.size(1), self.mem_size)
            ).to(self.DEVICE)

            # Iterate over number of frames
            for t in range(inputVariable.size(0)):
                # ResNet forward pass for each t frame of a video
                # spatial_frame_feat: (bs, 512, 7, 7)
                _, spatial_frame_feat, _ = self.resNet(inputVariable[t])

                # Average pooling for each t frame of a video
                # frames_feat: (bs, 512)
                frame_feat = self.avgpool(spatial_frame_feat).view(
                    spatial_frame_feat.size(0), -1
                )

                # Add to video level features
                video_level_features = video_level_features + frame_feat

            # Mean of the frames for the video
            video_level_features = video_level_features / inputVariable.size(0)

            # Classifier
            logits = self.classifier(video_level_features)
            return logits, video_level_features

        # Learning with Temporal information (LSTM)
        elif self.homework_step == 1:
            state = (
                torch.zeros((inputVariable.size(1), self.mem_size)).to(self.DEVICE),
                torch.zeros((inputVariable.size(1), self.mem_size)).to(self.DEVICE),
            )
            for t in range(inputVariable.size(0)):
                # spatial_frame_feat: (bs, 512, 7, 7)
                _, spatial_frame_feat, _ = self.resNet(inputVariable[t])
                # frames_feat: (bs, 512)
                frame_feat = self.avgpool(spatial_frame_feat).view(state[1].size(0), -1)
                state = self.lstm_cell(frame_feat, state)

            video_level_features = state[1]
            logits = self.classifier(video_level_features)
            return logits, video_level_features

        # Learning with Temporal information (ConvLSTM)
        elif self.homework_step == 2:
            state = (
                torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).to(
                    self.DEVICE
                ),
                torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).to(
                    self.DEVICE
                ),
            )
            for t in range(inputVariable.size(0)):
                # spatial_frame_feat: (bs, 512, 7, 7)
                _, spatial_frame_feat, _ = self.resNet(inputVariable[t])
                state = self.lstm_cell(spatial_frame_feat, state)
            video_level_features = self.avgpool(state[1]).view(state[1].size(0), -1)
            logits = self.classifier(video_level_features)
            return logits, video_level_features


def pipeline(
    homework_step, validate=True, steps=["train", "test"], model_name="model.pth"
):
    # CUDA_LAUNCH_BLOCKING=1
    print(DEVICE)

    model = ourModel(
        num_classes=NUM_CLASSES,
        mem_size=MEM_SIZE,
        homework_step=homework_step,
        DEVICE=DEVICE,
    )  # model

    # path to save model
    model_folder = os.path.join(
        CURRENT_DIR,
        "data/saved_models/" + "/" + "homework_step" + str(homework_step) + "/",
    )
    if not os.path.isdir(model_folder):
        os.makedirs(model_folder)

    # Train only the lstm cell and classifier
    model.train(False)
    for params in model.parameters():
        params.requires_grad = False

    if homework_step > 0:
        for params in model.lstm_cell.parameters():
            params.requires_grad = True
        model.lstm_cell.train(True)

    for params in model.classifier.parameters():
        params.requires_grad = True

    model.classifier.train(True)

    model = model.to(DEVICE)

    # Loss
    loss_fn = nn.CrossEntropyLoss()
    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_fn = optim.Adam(
        trainable_params, lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-4
    )
    # Scheduler
    optim_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer_fn, milestones=STEP_SIZE, gamma=GAMMA
    )

    train_loader, val_loader, train_size, test_size = load_data()
    for phase in steps:
        if phase == "train":
            train(
                model,
                loss_fn,
                optimizer_fn,
                optim_scheduler,
                homework_step,
                train_loader,
                val_loader,
                train_size,
                test_size,
                model_folder,
                validate=validate,
            )
        else:
            model.load_state_dict(torch.load(os.path.join(model_folder, model_name)))
            test(model, loss_fn, val_loader, test_size)


###### Training ######
def train(
    model,
    loss_fn,
    optimizer_fn,
    scheduler,
    homework_step,
    train_loader,
    val_loader,
    train_size,
    test_size,
    model_folder,
    validate=True,
    model_name="model.pth",
):
    train_iter = 0
    val_iter = 0
    min_accuracy = 0

    trainSamples = train_size - (train_size % BATCH_SIZE)
    val_samples = test_size
    iterPerEpoch = len(train_loader)
    val_steps = len(val_loader)
    cudnn.benchmark

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        numCorrTrain = 0

        # blocks to train
        if homework_step > 0:
            model.lstm_cell.train(True)
        model.classifier.train(True)

        for i, (inputs, targets) in enumerate(train_loader):
            train_iter += 1
            optimizer_fn.zero_grad()

            # (BS, Frames, C, W, H) --> (Frames, BS, C, W, H)
            inputVariable = inputs.permute(1, 0, 2, 3, 4).to(DEVICE)
            labelVariable = targets.to(DEVICE)

            # feeds in model
            output_label, _ = model(inputVariable)

            # compute loss
            loss = loss_fn(output_label, labelVariable)

            # backward loss and optimizer step
            loss.backward()
            optimizer_fn.step()

            # compute the training accuracy
            _, predicted = torch.max(output_label.data, 1)
            numCorrTrain += torch.sum(predicted == labelVariable.data).data.item()
            step_loss = loss.data.item()
            epoch_loss += step_loss

        avg_loss = epoch_loss / iterPerEpoch
        trainAccuracy = (numCorrTrain / trainSamples) * 100
        # train_logger.add_epoch_data(epoch+1, trainAccuracy, avg_loss)
        print(
            Fore.BLACK
            + "Train: Epoch = {} | Loss = {:.3f} | Accuracy = {:.3f}".format(
                epoch + 1, avg_loss, trainAccuracy
            )
        )
        if validate:
            if (epoch + 1) % 1 == 0:
                model.train(False)
                val_loss_epoch = 0
                numCorr = 0
                for j, (inputs, targets) in enumerate(val_loader):
                    val_iter += 1
                    inputVariable = inputs.permute(1, 0, 2, 3, 4).to(DEVICE)
                    labelVariable = targets.to(DEVICE)

                    output_label, _ = model(inputVariable)
                    val_loss = loss_fn(output_label, labelVariable)
                    val_loss_step = val_loss.data.item()
                    val_loss_epoch += val_loss_step
                    _, predicted = torch.max(output_label.data, 1)
                    numCorr += torch.sum(predicted == labelVariable.data).data.item()
                    # val_logger.add_step_data(val_iter, numCorr, val_loss_step)

                val_accuracy = (numCorr / val_samples) * 100
                avg_val_loss = val_loss_epoch / val_steps

                print(
                    Fore.GREEN
                    + "Val: Epoch = {} | Loss {:.3f} | Accuracy = {:.3f}".format(
                        epoch + 1, avg_val_loss, val_accuracy
                    )
                )
                if val_accuracy > min_accuracy:
                    print("[||| NEW BEST on val||||]")
                    save_path_model = os.path.join(model_folder, model_name)
                    torch.save(model.state_dict(), save_path_model)
                    min_accuracy = val_accuracy

        scheduler.step()

    print(Fore.CYAN + "Best Acc --> ", min_accuracy)
    print(Fore.CYAN + "Last Acc --> ", val_accuracy)


def test(model, loss_fn, val_loader, test_size):
    model.train(False)
    val_loss_epoch = 0
    numCorr = 0
    val_iter = 0
    val_samples = test_size
    val_steps = len(val_loader)

    with torch.no_grad():
        for j, (inputs, targets) in enumerate(val_loader):
            val_iter += 1
            inputVariable = inputs.permute(1, 0, 2, 3, 4).to(DEVICE)
            labelVariable = targets.to(DEVICE)

            output_label, _ = model(inputVariable)
            val_loss = loss_fn(output_label, labelVariable)
            val_loss_step = val_loss.data.item()
            val_loss_epoch += val_loss_step
            _, predicted = torch.max(output_label.data, 1)
            numCorr += torch.sum(predicted == labelVariable.data).data.item()

        val_accuracy = (numCorr / val_samples) * 100
        avg_val_loss = val_loss_epoch / val_steps

    print("Loss {:.3f} | Accuracy = {:.3f}".format(avg_val_loss, val_accuracy))


if __name__ == "__main__":
    # Homework step
    # homework_step = 0  # --> Learning without Temporal information (avgpool)
    # homework_step = 1 #--> Learning with Temporal information (LSTM)
    # homework_step = 2 #--> Learning with Spatio-Temporal information (ConvLSTM)
    pipeline(
        homework_step=1, validate=True, steps=["train", "test"], model_name="model1.pth"
    )
