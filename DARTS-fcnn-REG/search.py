""" Search cell """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import SearchConfig
import utils
from models.search_fcnn import SearchFCNNController
from architect import Architect
from visualize import plot
import random
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline

config = SearchConfig()

# device = torch.device("cuda")
device = torch.device("cpu")

# tensorboard
print("log_dir=", "." + os.path.join(config.path, "tb"))
writer = SummaryWriter(log_dir="." + os.path.join(config.path, "tb"))
writer.add_text("config", config.as_markdown(), 0)

logger = utils.get_logger("." + os.path.join(config.path, "{}.log".format(config.name)))
config.print_params(logger.info)


def minmaxstdscaler():
    """Use MinMaxScaler then StandardScaler.
    Returns:
        preprocessor:
    """

    preprocessor = Pipeline(
        [("minmaxscaler", MinMaxScaler()), ("stdscaler", StandardScaler())]
    )
    return preprocessor


def linear_():
    p = lambda x: sum(x)
    a = 100
    b = 1
    minimas = lambda d: [-1 for i in range(d)]
    return p, (a, b), minimas


def polynome_2():
    p = lambda x: -sum([x_i ** 2 for x_i in x])
    a = -2
    b = 2
    minimas = lambda d: [0 for i in range(d)]
    return p, (a, b), minimas


def load_data(dim=2):
    """
    Generate data for linear function -sum(x_i).
    Return:
        Tuple of Numpy arrays: ``(train_X, train_y), (valid_X, valid_y)``.
    """
    n_samples = 1000
    test_size = 0.33
    # f, (a, b), _ = linear_()
    f, (a, b), _ = polynome_2()
    d = b - a
    X = np.array(
        [a + np.random.random(dim) * d for i in range(n_samples)], dtype=np.float32
    )
    y = np.array([[f(v)] for v in X], dtype=np.float32)

    train_X, valid_X, train_y, valid_y = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    preproc = minmaxstdscaler()
    train_X = preproc.fit_transform(train_X)
    valid_X = preproc.transform(valid_X)

    print(f"train_X shape: {np.shape(train_X)}")
    print(f"train_y shape: {np.shape(train_y)}")
    print(f"valid_X shape: {np.shape(valid_X)}")
    print(f"valid_y shape: {np.shape(valid_y)}")
    return (train_X, train_y), (valid_X, valid_y)


def main():
    logger.info("Logger is set - training start")

    # set default gpu device id
    # torch.cuda.set_device(config.gpus[0])

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    # torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = False

    # get data with meta info
    (train_X, train_y), (valid_X, valid_y) = load_data()
    in_dim = np.shape(train_X)[1]
    out_dim = np.shape(train_y)[1]
    # out_dim = train_y.max() + 1

    train_X, train_y = (torch.tensor(train_X, dtype=torch.float), torch.tensor(train_y))
    train_data = torch.utils.data.TensorDataset(train_X, train_y)

    valid_X, valid_y = (torch.tensor(valid_X, dtype=torch.float), torch.tensor(valid_y))
    valid_data = torch.utils.data.TensorDataset(valid_X, valid_y)
    print("in_dim: ", in_dim)
    print("out_dim: ", out_dim)

    net_crit = nn.MSELoss().to(device)
    layers = 1
    n_nodes = 4
    model = SearchFCNNController(
        in_dim, out_dim, layers, net_crit, n_nodes=n_nodes, device_ids=config.gpus
    )
    model = model.to(device)

    # weights optimizer
    # w_optim = torch.optim.SGD(
    #     model.weights(),
    #     config.w_lr,
    #     momentum=config.w_momentum,
    #     weight_decay=config.w_weight_decay,
    # )
    w_optim = torch.optim.RMSprop(model.weights())
    # alphas optimizer
    # alpha_lr = config.alpha_lr
    alpha_lr = 0.01
    alpha_optim = torch.optim.Adam(
        model.alphas(),
        alpha_lr,
        betas=(0.5, 0.999),
        # weight_decay=config.alpha_weight_decay,
    )

    # split data to train/validation
    n_train = len(train_data)
    n_valid = len(valid_data)
    indices_train = list(range(n_train))
    indices_valid = list(range(n_valid))
    random.shuffle(indices_train)
    random.shuffle(indices_valid)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices_train)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices_valid)
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.workers,
        pin_memory=True,
    )
    # vis.
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    # writer.add_graph(model, [images[0]])
    # writer.close()

    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=config.batch_size,
        sampler=valid_sampler,
        num_workers=config.workers,
        pin_memory=True,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        w_optim, config.epochs, eta_min=config.w_lr_min
    )
    architect = Architect(model, config.w_momentum, config.w_weight_decay)

    # training loop
    best_top1 = None
    epochs = config.epochs
    for epoch in range(epochs):
        lr = lr_scheduler.get_lr()[0]

        model.print_alphas(logger)

        # training
        train(
            train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch
        )
        lr_scheduler.step()

        # validation
        cur_step = (epoch + 1) * len(train_loader)
        top1 = validate(valid_loader, model, epoch, cur_step)

        # log
        # genotype
        genotype = model.genotype()
        logger.info("genotype = {}".format(genotype))

        # genotype as a image
        plot_path = "." + os.path.join(config.plot_path, "EP{:02d}".format(epoch + 1))
        caption = "Epoch {}".format(epoch + 1)
        plot(genotype.normal, plot_path + "-normal", caption)

        # save
        if best_top1 is None or best_top1 < top1:
            best_top1 = top1
            best_genotype = genotype
            is_best = True
        else:
            is_best = False
        # utils.save_checkpoint(model, "." + config.path, is_best)
        print("")

    best_genotype = model.genotype()

    with open("." + config.path + "/best_genotype.txt", "w") as f:
        f.write(str(best_genotype))

    logger.info("Final best TopR2@1 = {:.3f}".format(best_top1))
    print("best_top1: ", best_top1)
    logger.info("Best Genotype = {}".format(best_genotype))


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    cur_step = epoch * len(train_loader)
    writer.add_scalar("train/lr", lr, cur_step)

    model.train()

    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(
        zip(train_loader, valid_loader)
    ):
        trn_X, trn_y = (
            trn_X.to(device, non_blocking=True),
            trn_y.to(device, non_blocking=True),
        )
        val_X, val_y = (
            val_X.to(device, non_blocking=True),
            val_y.to(device, non_blocking=True),
        )
        N = trn_X.size(0)

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.weights(), config.w_grad_clip)
        w_optim.step()

        r2score = utils.r2score(logits, trn_y)

        losses.update(loss.item(), N)
        top1.update(r2score, N)

        if step % config.print_freq == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} ".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses
                )
            )

        writer.add_scalar("train/loss", loss.item(), cur_step)
        writer.add_scalar("train/top1-r2score", r2score, cur_step)

        cur_step += 1

    logger.info(
        "Train: [{:2d}/{}] Final Loss {:.3f}".format(epoch + 1, config.epochs, losses.avg)
    )


def validate(valid_loader, model, epoch, cur_step):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            N = X.size(0)

            logits = model(X)
            loss = model.criterion(logits, y)
            r2score = utils.r2score(logits, y)

            losses.update(loss.item(), N)
            top1.update(r2score, N)

            if step % config.print_freq == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} ".format(
                        epoch + 1,
                        config.epochs,
                        step,
                        len(valid_loader) - 1,
                        losses=losses,
                    )
                )

    writer.add_scalar("val/loss", losses.avg, cur_step)
    writer.add_scalar("val/top1-r2score", top1.avg, cur_step)

    logger.info(
        "Valid: [{:2d}/{}] Final Loss {:.3f}".format(epoch + 1, config.epochs, losses.avg)
    )

    return top1.avg


if __name__ == "__main__":
    main()
