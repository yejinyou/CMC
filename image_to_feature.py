import argparse
import matplotlib.pyplot as plt
from models.resnet import InsResNet50
import numpy as np
import os, sys, socket
import torch
from torchvision import transforms, datasets
from quickdataset import QuickImageFolder
import pickle


def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--crop_pad', type=int, default=32, help='crop padding')
    parser.add_argument('--imsize', type=int, default=224, help='image size')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=24, help='num of workers to use')
    parser.add_argument('--epochs', type=int, nargs='+', default=200, help='model trained epochs')
    parser.add_argument('--folder', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'places365'])
    
    opt = parser.parse_args()

    return opt


def save_obj(obj, name ):
    print("==> Saving dictionary to ... " + name) 
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def img_to_feat(model, val_loader):
    print("==> Converting images to features")
    idx_to_feat = {}
    #idx_to_img = {}
    #idx_to_label = {}
    
    n = len(val_loader)
    for i, (img, labels, idx) in enumerate(val_loader):
        if i % 1000 == 0: print("iter [{}/{}]".format(i, n))
        idx = idx.detach().numpy()[0]
        x1, x2 = torch.split(img, [3, 3], dim=1)
        idx_to_feat[idx] = model(x1).squeeze().cpu().detach().numpy()
        #idx_to_img[idx] = x1.squeeze().cpu().detach().numpy()
        #idx_to_label[idx] = labels.detach().numpy()[0]
    return idx_to_feat#, idx_to_label


def main():
    
    args = parse_option()
    args.moco = True
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    my_path = "/data/vision/torralba/ganprojects/yyou/CMC"
    
    model_name = "/{}_MoCo0.999_softmax_16384_resnet50_lr_0.03_decay_0.0001_bsz_128_crop_0.2_aug_CJ".format(args.dataset)
    
    if args.dataset == "imagenet":
        dataset_path = "/data/vision/torralba/datasets/imagenet_pytorch/imagenet_pytorch"
        model_ckpt_path = "/imagenet_models"
    elif args.dataset == "places365":
        dataset_path = "/data/vision/torralba/datasets/places/places365_standard/places365standard_easyformat"
        model_ckpt_path = "/places365_models"
    else:
        raise ValueError("Unsupported dataset type of {}".format(args.datset))

    train_folder = dataset_path + "/train"
    val_folder = dataset_path + "/val"
        
    if args.folder == "train":
        print("=> Loading train set")
        crop = 0.8
        train_sampler = None

        transform = transforms.Compose([
                        transforms.RandomResizedCrop(args.imsize, scale=(crop, 1.)),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ])
        dataset = QuickImageFolder(train_folder, transform=transform, two_crop=args.moco)
        loader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                    num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    else:
        print("=> Loading val set")
        transform = transforms.Compose([
                            transforms.Resize(args.imsize + args.crop_pad),
                            transforms.CenterCrop(args.imsize),
                            transforms.ToTensor(),
                            normalize,
                        ])

        dataset = QuickImageFolder(val_folder, transform=transform, two_crop=args.moco)
        loader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size, 
                    shuffle=False, num_workers=args.num_workers, pin_memory=True)
            
    if not isinstance(args.epochs, list):
        args.epochs = [args.epochs]

    print("Generating for epochs: ", args.epochs)
    
    for epoch in args.epochs:
        print("====> Working on epoch: {}".format(epoch))
        epoch_name = "/ckpt_epoch_{}.pth".format(epoch)

        model_path = my_path + model_ckpt_path + model_name + epoch_name

        model = InsResNet50()
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        model.cuda()

        #idx_to_feat, idx_to_label= img_to_feat(model, loader)
        idx_to_feat = img_to_feat(model, loader)
        save_obj(idx_to_feat, my_path+"/pkl"+"/{}_{}_to_feat_epoch{}".format(args.dataset, args.folder, epoch))
        #save_obj(idx_to_img, my_path+"/pkl"+"/{}_to_img_epch{}".format(args.folder, args.epoch))
#         save_obj(idx_to_label, my_path+"/pkl"+"/{}_{}_to_label_epoch{}".format(args.dataset, args.folder, epoch))

    
    
if __name__ == "__main__":
    main()
    
