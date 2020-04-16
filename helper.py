from CMC.quickdataset import QuickImageFolder
from CMC.models.resnet import InsResNet50
from netdissect import nethook
import torch
from torchvision import transforms


crop = 0.2
crop_padding = 32

num_workers = 24
train_sampler = None
moco = True

image_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)



def get_val_dataset(dataset, batch_size=1):

    if dataset == "imagenet":
        data_path =  "/data/vision/torralba/datasets/imagenet_pytorch/imagenet_pytorch"
    elif dataset == "places365":
        data_path = "/data/vision/torralba/datasets/places/" + \
                    "places365_standard/places365standard_easyformat"
    val_folder = data_path + "/val"

    val_transform = transforms.Compose([
                    transforms.Resize(image_size + crop_padding),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    normalize,
                ])
    
    ds = QuickImageFolder(val_folder, transform=val_transform, shuffle=True)#, two_crop=False)
    ds_loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    return ds, ds_loader


def get_moco_model(dataset, epoch=240):
    
    folder_path = "CMC/CMC_data/{}_models".format(dataset)
    model_name = "/{}_MoCo0.999_softmax_16384_resnet50".format(dataset) + \
                 "_lr_0.03_decay_0.0001_bsz_128_crop_0.2_aug_CJ"
    epoch_name = "/ckpt_epoch_{}.pth".format(epoch)
    my_path = folder_path + model_name + epoch_name
    
    checkpoint = torch.load(my_path)
    model_checkpoint = {key.replace(".module",""):val for key, val in checkpoint['model'].items()}

    model = InsResNet50(parallel=False)
    model.load_state_dict(model_checkpoint)
    model = nethook.InstrumentedModel(model)
    return model


