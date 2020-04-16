from quickdataset import QuickImageFolder
from models.resnet import InsResNet50
from image_helper import *
import IPython
from netdissect import nethook, imgviz, show, segmenter, renormalize, upsample, tally, pbar, setting
import torch, os, matplotlib.pyplot as plt
from torchvision import transforms


def parse_option():

    hostname = socket.gethostname()
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--sample_size', type=int, default=1000, help='sample size') 
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'places365'])
    parser.add_argument('--layer', type=str, default='')

    opt = parser.parse_args()

    return opt


def main():
    
    # Load the arguments
    args = parse_option()
    
    dataset = args.dataset
    sample_size = args.sample_size
    layername = args.layer
    
    # Other values for places and imagenet MoCo model
    epoch = 240
    image_size = 224
    crop = 0.2
    crop_padding = 32
    batch_size = 1
    num_workers = 24
    train_sampler = None
    moco = True

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    
    
    # Set appropriate paths
    folder_path = "/data/vision/torralba/ganprojects/yyou/CMC_data/{}_models".format(dataset)
    model_name = "/{}_MoCo0.999_softmax_16384_resnet50_lr_0.03".format(dataset) \
                     + "_decay_0.0001_bsz_128_crop_0.2_aug_CJ"
    epoch_name = "/ckpt_epoch_{}.pth".format(epoch)
    my_path = folder_path + model_name + epoch_name
    
    data_path = "/data/vision/torralba/datasets/"
    web_path = "/data/vision/torralba/scratch/yyou/wednesday/dissection/"
    
    if dataset == "imagenet":
        data_path +=  "imagenet_pytorch"
        web_path += dataset + "/" + layername
    elif dataset == "places365":
        data_path += "places/places365_standard/places365standard_easyformat"
        web_path += dataset + "/" + layername

    # Create web path folder directory for this layer
    if not os.path.exists(web_path):
        os.makedirs(web_path)
    
    # Load validation data loader
    val_folder = data_path + "/val"
    val_transform = transforms.Compose([
                        transforms.Resize(image_size + crop_padding),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        normalize,
                    ])

    ds = QuickImageFolder(val_folder, transform=val_transform, shuffle=True, two_crop=False)
    ds_loader = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=train_sampler)

    
    # Load model from checkpoint
    checkpoint = torch.load(my_path)
    model_checkpoint = {key.replace(".module",""):val for key, val in checkpoint['model'].items()}

    model = InsResNet50(parallel=False)
    model.load_state_dict(model_checkpoint)
    model = nethook.InstrumentedModel(model)
    model.cuda()

    # Renormalize RGB data from the staistical scaling in ds to [-1...1] range
    renorm = renormalize.renormalizer(source=ds, target='zc')

    
    # Retain desired layer with nethook
    print("===> Retaining desired layer ... ", layername)
    batch = next(iter(ds_loader))[0]
    model.retain_layer(layername)
    model(batch.cuda())
    acts = model.retained_layer(layername).cpu()

    upfn = upsample.upsampler(
        target_shape=(56, 56),
        data_shape=(7, 7),
    )

    def flatten_activations(batch, *args):
        image_batch = batch
        _ = model(image_batch.cuda())
        acts = model.retained_layer(layername).cpu()
        hacts = upfn(acts)
        return hacts.permute(0, 2, 3, 1).contiguous().view(-1, acts.shape[1])

    def tally_quantile_for_layer(layername):
        rq = tally.tally_quantile(
            flatten_activations,
            dataset=ds,
            sample_size=sample_size,
            batch_size=100,
            cachefile='results/{}/{}_rq_cache.npz'.format(dataset, layername))
        return rq

    rq = tally_quantile_for_layer(layername)
    
    
    # Visualize range of activations (statistics of each filter over the sample images)
    print("===> Visualizing range of activations ... ")
    fig, axs = plt.subplots(2, 2, figsize=(10,8))
    axs = axs.flatten()
    quantiles = [0.5, 0.8, 0.9, 0.99]
    for i in range(4):
        axs[i].plot(rq.quantiles(quantiles[i]))
        axs[i].set_title("Rq quantiles ({})".format(quantiles[i]))
    fig.suptitle("{}  -  sample size of {}".format(dataset, sample_size))
    plt.savefig(web_path + "/rq_quantiles")
    print("===> Output written to ... {}".format(web_path)

    
    # Set the image visualizer with the rq and percent level
    iv = imgviz.ImageVisualizer(224, source=ds, percent_level=0.95, quantiles=rq)

    
    # Tally top k images that maximize the mean activation of the filter
    def max_activations(batch, *args):
        image_batch = batch.cuda()
        _ = model(image_batch)
        acts = model.retained_layer(layername)
        return acts.view(acts.shape[:2] + (-1,)).max(2)[0]

    def mean_activations(batch, *args):
        image_batch = batch.cuda()
        _ = model(image_batch)
        acts = model.retained_layer(layername)
        return acts.view(acts.shape[:2] + (-1,)).mean(2)

    topk = tally.tally_topk(
        mean_activations,
        dataset=ds,
        sample_size=sample_size,
        batch_size=100,
        cachefile='results/{}/{}_cache_mean_topk.npz'.format(dataset, layername)
    )

    top_indexes = topk.result()[1]

    
    # Visualize top-activating images for a particular unit
    print("===> Visualizing top-activated images for units ... ")
    indices = np.random.randint(sample_size, size=10)
    
    def top_activating_imgs(unit):
        img_ids = [i for i in top_indexes[unit, :12]]
        images = [iv.masked_image(ds[i][0], \
                      model.retained_layer(layername)[0], u) \
                      for i in img_ids]
        preds = [ds.classes[model(ds[i][0][None].cuda()).max(1)[1].item()]\
                    for i in img_ids]

        fig, axs = plt.subplots(3, 4, figsize=(16, 12))
        axs = axs.flatten()

        for i in range(12):
            axs[i].imshow(images[i])
            axs[i].tick_params(axis='both', which='both', bottom=False, \
                               left=False, labelbottom=False, labelleft=False)
            axs[i].set_title("img {} \n pred: {}".format(img_ids[i], preds[i]))
        fig.suptitle("unit {}".format(unit))


        plt.savefig(web_path + "/top_activating_imgs/unit_{}".format(unit))

    for unit in indices:
        top_activating_imgs(unit)
    
    
#     def compute_activations(image_batch):
#         image_batch = image_batch.cuda()
#         _ = model(image_batch)
#         acts_batch = model.retained_layer(layername)
#         return acts_batch

#     unit_images = iv.masked_images_for_topk(
#         compute_activations,
#         ds,
#         topk,
#         k=5,
#         num_workers=10,
#         pin_memory=True,
#         cachefile='results/{}/{}_cache_top10images.npz'.format(dataset, layername))


    # Load the segmentation model 
    segmodel, seglabels, segcatlabels = setting.load_segmenter('netpqc')

    # Intersection between every unit's 99th activation
    # and every segmentation class identified by the semgenter
    level_at_99 = rq.quantiles(0.99).cuda()[None,:,None,None]

    def compute_selected_segments(batch, *args):
        image_batch = batch.cuda()
        seg = segmodel.segment_batch(renorm(image_batch), downsample=4)
        _ = model(image_batch)
        acts = model.retained_layer(layername)
        hacts = upfn(acts)
        iacts = (hacts > level_at_99).float() # indicator where > 0.99 percentile.
        return tally.conditional_samples(iacts, seg)

    condi99 = tally.tally_conditional_mean(
        compute_selected_segments,
        dataset=ds,
        sample_size=sample_size,
        cachefile='results/{}/{}_cache_condi99.npz'.format(dataset, layername))

    iou99 = tally.iou_from_conditional_indicator_mean(condi99)
    iou99.shape


    # Units with the best match to a segmentation class 
    iou_unit_label_99 = sorted([(
        unit, concept.item(), seglabels[concept], bestiou.item())
        for unit, (bestiou, concept) in enumerate(zip(*iou99.max(0)))],
        key=lambda x: -x[-1])
    
    print("===> Visualizing best unit segmentations ... ")
    fig, axs = plt.subplots(20, 1, figsize=(20,80))
    axs = axs.flatten()

    for i, (unit, concept, label, score) in enumerate(iou_unit_label_99[:20]):
        axs[i].imshow(unit_images[unit])
        axs[i].set_title('unit %d; iou %g; label "%s"' % (unit, score, label))
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.savefig(web_path + "/best_unit_segmentation")


if __name__ == "__main__":
    
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    
    main()
