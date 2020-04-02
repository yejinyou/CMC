import torch 
from netdissect import show, renormalize

def imsave(imlist, fname, source='imagenet', blocks=5):
    """
    Save the list of images(pixel tensors) to the inputted filename
    """
    if len(imlist)==1:
        renormalize.as_image(imlist[0], source=source).save(fname, format='png')
    else:
        im1 = renormalize.as_image(imlist[0], source=source)
        w, h = im1.width, im1.height
        num_vert = len(imlist) // blocks + 1
        dst = Image.new('RGB', (w*blocks, h*num_vert))
        for i, img in enumerate(imlist):
            dst.paste(renormalize.as_image(img, source=source),\
                      ((i%blocks)*w, (i//blocks)*h))
        dst.save(fname, format='png')

def split_img(img):
    x1, x2 = torch.split(img, [3, 3], dim=0)
    return x1

def show_idx(idxlist, source='imagenet'):
    """
    Show images corresponding to the inputted index list
    """
    if len(idxlist)==1:
        print("image", idxlist[0])
        show(renormalize.as_image(split_img(val_dataset[idxlist[0]][0]), source=source))
    else:
        print("idx list", idxlist)
        show([[renormalize.as_image(split_img(val_dataset[idx][0]), source=source)] for idx in idxlist])