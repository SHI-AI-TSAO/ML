from auto_vp.wrapper import BaseWrapper
from auto_vp.utilities import setup_device
from auto_vp.dataprepare import DataPrepare, Data_Scalability
from auto_vp import programs
from auto_vp.training_process import Training
from auto_vp.const import CLASS_NUMBER, IMG_SIZE, SOURCE_CLASS_NUM, BATCH_SIZE, NETMEAN, NETSTD, DEFAULT_TEMPLATE, ENSEMBLE_TEMPLATES
from auto_vp.load_model import Load_Reprogramming_Model

import argparse
from torchvision import transforms
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import manifold
from torch.nn import functional as F
import cv2
import os
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import torchvision
from functools import partial

# Package
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True

# https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
class Network(nn.Module):
    def __init__(self, reprogram_model):
        super(Network, self).__init__()
        
        #print(reprogram_model)
        # get the pretrained network
        self.reprogram_model = reprogram_model
        self.pretrained_name = reprogram_model.model_name
        self.pretrained_model = reprogram_model.model
        print(self.pretrained_model)
        
        # disect the network to access its last convolutional layer
        #if(self.pretrained_name == "clip"):
    
        if(self.pretrained_name == "resnet18"):
            self.features_layer = nn.Sequential(
                self.pretrained_model.conv1,
                self.pretrained_model.bn1,
                self.pretrained_model.relu,
                self.pretrained_model.maxpool,

                self.pretrained_model.layer1,
                self.pretrained_model.layer2,
                self.pretrained_model.layer3,
                self.pretrained_model.layer4,
                
                )

            self.rest_layer = nn.Sequential(
                
                self.pretrained_model.avgpool,
                nn.Flatten(start_dim=1), # torch.flatten(x, 1),
                self.pretrained_model.fc
                ) 
        self.gradients = None
    
    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # clip need to resize by ourself 
        x = self.reprogram_model.clip_rz_transform(x)

        img_h = -1
        img_w = -1
        if(self.reprogram_model.no_trainable_resize == 0):
            x, img_h, img_w = self.reprogram_model.train_resize(x)
        else:
            x = self.reprogram_model.train_resize(x)

        xp = self.reprogram_model.input_perturbation(x, img_h, img_w)

        if(self.pretrained_name[0:4] == "clip"):
            x = self.reprogram_model.CLIP_network(xp) ####
        elif(self.pretrained_name == "resnet18"):
            # x = self.model(x)
            # Step1: Feature extraction
            x = self.features_layer(xp)
            # Step2: register the hook
            h = x.register_hook(self.activations_hook)
            # Step3: apply the remaining pooling
            x = self.rest_layer(x)
        else:
            x = self.reprogram_model.model(xp)

        
        x = self.reprogram_model.output_mapping(x)
        
        return x, xp
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_layer(x)

# for CLIP
def interpret(image, texts, model, device, start_layer=-1, start_layer_text=-1):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1: 
        # calculate index of last layer 
        start_layer = len(image_attn_blocks) - 1
    
    print(image_attn_blocks[0])
    num_tokens = image_attn_blocks[0].attn.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]

    
    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1: 
        # calculate index of last layer 
        start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
            continue
        grad = torch.autograd.grad(one_hot, [blk.attn], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R_text = R_text + torch.bmm(cam, R_text)
    text_relevance = R_text

    return text_relevance, image_relevance

def show_image_relevance(image_relevance, image, orig_image, axs=None):
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    axs.imshow(orig_image)
    axs.axis('off')

def show_heatmap_on_text(text, text_encoding, R_text, axs=None):
    CLS_idx = text_encoding.argmax(dim=-1)
    R_text = R_text[CLS_idx, 1:CLS_idx]
    text_scores = R_text / R_text.sum()
    text_scores = text_scores.flatten()
    print(text_scores)
    text_tokens=_tokenizer.encode(text)
    text_tokens_decoded=[_tokenizer.decode([a]) for a in text_tokens]
    vis_data_records = [visualization.VisualizationDataRecord(text_scores,0,0,0,0,0,text_tokens_decoded,1)]
    visualization.visualize_text(vis_data_records)
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    axs.imshow(vis)
    axs.axis('off')

def get_last_selfattention(self, x):
    x = self.prepare_tokens(x)
    for i, blk in enumerate(self.blocks):
        if i < len(self.blocks) - 1:
            x = blk(x)
        else:
            # return attention of the last block
            return blk(x, return_attention=True)

def get_saparate_text_embedding(classnames, templates, model, device): #####@@@@@@@
    zeroshot_weights = []
    if isinstance(templates, list):
        for template in tqdm(templates, desc="Embedding texts", ncols=100):
            texts = [template.format(classname) for classname in classnames]
            texts = clip.tokenize(texts).to(device)
            with torch.no_grad():
                text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            zeroshot_weights.append(text_embeddings)
    else:
        texts = [templates.format(classname) for classname in classnames]
        texts = clip.tokenize(texts).to(device)
        with torch.no_grad():
            text_embeddings = model.encode_text(texts)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        zeroshot_weights = text_embeddings

    # print(self.text_content)
    return zeroshot_weights

def CLIP_Text_Embedding(class_names, template_number, clip_model, device):
    TEMPLATES = [DEFAULT_TEMPLATE] + ENSEMBLE_TEMPLATES # len(TEMPLATES): 81
    txt_emb = get_saparate_text_embedding(class_names, TEMPLATES[template_number], clip_model, device) 
    return txt_emb

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--dataset', choices=["CIFAR10", "CIFAR10-C", "CIFAR100", "ABIDE", "Melanoma", "DR", "SVHN", "GTSRB", "Flowers102", "DTD", "Food101", "EuroSAT", "OxfordIIITPet", "UCF101", "Camelyon17", "Iwildcam", "FMoW", "CLEVR_count"], default="GTSRB")
    p.add_argument('--datapath', type=str, default="/home/joytsao/Reprogramming/GTSRB")
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--scalibility_rio', type=int, choices=[1, 2, 4, 10, 100], default=1)
    p.add_argument('--scalibility_mode', choices=["equal", "random"], default="equal")  

    args = p.parse_args()

    # set random seed
    set_seed(args.seed)

    # device setting
    device, list_ids = setup_device(1)
    #os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
    device = 'cpu' #'cuda:1' #
    print("device: ", device)

    ##### Model Setting #####
    channel = 3
    img_resize = IMG_SIZE[args.dataset]
    class_num = CLASS_NUMBER[args.dataset]
    # set_train_resize = True
    random_state = args.seed
    ##### End of Setting #####    

    # Load model 
    reprogram_model = Load_Reprogramming_Model(args.dataset, device, file_path=f"{args.dataset}_last.pth") #f"{args.dataset}_last.pth" # , set_train_resize=set_train_resize
    # reprogram_model = Load_Reprogramming_Model(args.dataset, device, mapping_method="fully_connected_layer_mapping", pretrained_model="resnet18", mapping=None, scale=1.0)
    if(reprogram_model.no_trainable_resize == 1):
        set_train_resize = False
    else:
        set_train_resize = True


    print("Model Info:")
    print("set_train_resize: ", set_train_resize)
    print(reprogram_model.model_name)
    print(reprogram_model.output_mapping.mapping_method)


    if(reprogram_model.model_name[0:4] == "clip"):
        clip_transform = reprogram_model.clip_preprocess
    else:
        clip_transform = None

    scale = reprogram_model.init_scale
    print(scale)
    if(set_train_resize == False):
        # redefind image size
        img_resize = int(img_resize*scale)
        if(img_resize > 224):
            img_resize = 224
    print(img_resize)
    
    print(reprogram_model.output_mapping.self_definded_map)
    if(set_train_resize == True):
        print(reprogram_model.train_resize.scale)

    wild_ds_list = ["Camelyon17", "Iwildcam", "FMoW"]
    if args.dataset in wild_ds_list:
        wild_dataset = True
    else:
        wild_dataset = False
            
    CIFAR10_C_mode = None

    # dataloader
    trainloader, testloader, class_names, trainset = DataPrepare(dataset_name=args.dataset, dataset_dir=args.datapath, target_size=(
        img_resize, img_resize), mean=NETMEAN[reprogram_model.model_name], std=NETSTD[reprogram_model.model_name], download=True, batch_size=BATCH_SIZE[args.dataset], random_state=random_state, clip_transform=clip_transform, CIFAR10_C_mode=CIFAR10_C_mode)

    if(args.scalibility_rio != 1):
        trainloader = Data_Scalability(trainset, args.scalibility_rio, BATCH_SIZE[args.dataset], mode=args.scalibility_mode, random_state=random_state, wild_dataset=wild_dataset) 
    # Plot t_SNE result    

    # Prepare text embedding
    if(reprogram_model.model_name[0:4] == "clip"):
        template_number = 0 # use default template
        reprogram_model.CLIP_Text_Embedding(class_names, template_number) 

    reprogram_model.eval()
    reprogram_model.model.requires_grad_(True)
    net = Network(reprogram_model)
    
    pbar = tqdm(testloader, total=len(testloader), desc=f"Testing", ncols=100)
    target_layers = [reprogram_model.model.layer1[-1]]  #[reprogram_model.model.visual.transformer.resblocks[-2].ln_2]#[reprogram_model.model.visual.transformer.resblocks[-1].ln_2]  #[reprogram_model.model.norm]
    mm = nn.Softmax(dim=1)
    #'''
    fig, ax = plt.subplots(2, 4, figsize=(20,10))
    ii = 0
    jj = 0
    for pb in pbar:
        if(jj < 5):
            jj += 1
            continue
        if(wild_dataset == True):
            imgs, labels, _ = pb
        else:
            imgs, labels = pb

        choose_target = ii + 7
        im = torch.unsqueeze(imgs[choose_target], 0)
        logits, purb_img = net(im.to(device))
        logits = mm(logits)
        pred_y = logits.argmax(dim=-1)
        confidence = logits[0, pred_y]


        input_tensor = torch.unsqueeze(imgs[choose_target], 0) # Create an input tensor image for your model..
        # Note: input_tensor can be a batch tensor with several images!

        # Construct the CAM object once, and then re-use it on many images:
        # with torch.autograd.set_detect_anomaly(True):
        cam = GradCAM(model=reprogram_model, target_layers=target_layers, use_cuda=False, reshape_transform=None)

        # If targets is None, the highest scoring category (for every member in the batch) will be used.
        targets =  [ClassifierOutputTarget(pred_y)]   #[ClassifierOutputTarget(254)] #

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        # print("@@@ input_tensor: ", input_tensor.size(), "targets: ", targets)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        del cam

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = torch.tensor(grayscale_cam[0, :])
        heatmap = np.maximum(grayscale_cam, 0)
        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # draw the heatmap
        img = purb_img[0]
        #print(img.shape)
        heatmap =  torch.unsqueeze(torch.unsqueeze(heatmap, 0),0)
        #print(heatmap.shape)
        m = nn.Upsample(size=(224,224), mode='bilinear')
        heatmap = m(heatmap)
        #print(heatmap.shape)

        ax[0][ii].imshow(np.transpose(np.float32(purb_img[0].detach().numpy()), (1, 2, 0)))
        ax[0][ii].set_aspect('equal', adjustable='box')
        im = ax[1][ii].imshow(np.transpose(img.cpu().detach().numpy(), (1, 2, 0)), alpha=0.4)
        ax[1][ii].contourf(heatmap[0,0], alpha=0.4, cmap=plt.cm.jet)
        ax[1][ii].set_aspect('equal', adjustable='box')
        ax[1][ii].set_title("y=" + class_names[labels[choose_target]] + ", pred_y=" + class_names[pred_y] + "\nconfidence=" + str(confidence.cpu().detach().numpy()))
    
        ii += 1
        if ii == 4:
            break

    plt.savefig("grad-cam-package")
    print("save grad-cam-package.png")

    #'''
    #'''
    ####### LP Model ########
    print("LP Model")
    pretrained = "ig_resnext101_32x8d" #"resnet18" #"clip" #
    
    state_dict = torch.load(args.dataset+"_LP_1_1.pth",  map_location=torch.device('cpu'))
    LP_model = state_dict['model']
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        transforms.ToTensor(),
        transforms.Normalize(NETMEAN[pretrained], NETSTD[pretrained]),
    ])
    target_layers = [LP_model.layer1[-1]]  #[LP_model.model.norm]

    LP_model.eval()
    LP_model.requires_grad_(True)

    if(pretrained[0:4] == "clip"):
        clip_transform = clip_preprocess
    else:
        clip_transform = preprocess

    # dataloader
    trainloader, testloader, class_names, trainset = DataPrepare(dataset_name=args.dataset, dataset_dir=args.datapath, target_size=(
        img_resize, img_resize), mean=NETMEAN[pretrained], std=NETSTD[pretrained], download=False, batch_size=BATCH_SIZE[args.dataset], random_state=random_state, clip_transform=clip_transform)
    
    pbar = tqdm(testloader, total=len(testloader), desc=f"Testing", ncols=100)
    
    fig, ax = plt.subplots(2, 4, figsize=(20,10))
    ii = 0
    jj = 0
    for pb in pbar:
        if(jj < 5):
            jj += 1
            continue
        if(wild_dataset == True):
            imgs, labels, _ = pb
        else:
            imgs, labels = pb

        choose_target = ii + 7
        im = torch.unsqueeze(imgs[choose_target], 0)
        logits = LP_model(im.to(device))
        '''
        with torch.no_grad():
            logits = LP_model(im.to(device))
        '''
        logits = mm(logits)
        pred_y = logits.argmax(dim=-1)
        confidence = logits[0, pred_y]
        print("pred_y: ", pred_y, "confidence: ", confidence)


        input_tensor = torch.unsqueeze(imgs[choose_target], 0) # Create an input tensor image for your model..
        # Note: input_tensor can be a batch tensor with several images!

        # Construct the CAM object once, and then re-use it on many images:
        cam = GradCAM(model=LP_model, target_layers=target_layers, use_cuda=False, reshape_transform=None) #reshape_transform

        targets = [ClassifierOutputTarget(pred_y)]  #[ClassifierOutputTarget(254)] # None

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        # print("@@@ input_tensor: ", input_tensor.size(), "targets: ", targets)
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        del cam

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = torch.tensor(grayscale_cam[0, :])
        
        heatmap = np.maximum(grayscale_cam, 0)
        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # draw the heatmap
        heatmap =  torch.unsqueeze(torch.unsqueeze(heatmap, 0),0)
        print(heatmap.shape)
        m = nn.Upsample(size=(224,224), mode='bilinear')
        heatmap = m(heatmap)
        print(heatmap.shape)

        ax[0][ii].imshow(np.transpose(np.float32(imgs[choose_target]), (1, 2, 0)))
        ax[0][ii].set_aspect('equal', adjustable='box')
        im = ax[1][ii].imshow(np.transpose(np.float32(imgs[choose_target]), (1, 2, 0)), alpha=0.4)
        ax[1][ii].contourf(heatmap[0,0], alpha=0.4, cmap=plt.cm.jet)
        ax[1][ii].set_aspect('equal', adjustable='box')
        ax[1][ii].set_title("y=" + class_names[labels[choose_target]] + ", pred_y=" + class_names[pred_y] + "\nconfidence=" + str(confidence.cpu().detach().numpy()))
    
        ii += 1
        if ii == 4:
            break

    plt.savefig("grad-cam-package-LP")
    print("save grad-cam-package-LP.png")
    #'''




    