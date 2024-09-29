from auto_reprogramming.wrapper import BaseWrapper
from auto_reprogramming.utilities import setup_device
from auto_reprogramming.dataprepare import DataPrepare, Data_Scalability
from auto_reprogramming import programs
from auto_reprogramming.training_process import Training
from auto_reprogramming.const import CLASS_NUMBER, IMG_SIZE, SOURCE_CLASS_NUM, BATCH_SIZE, NETMEAN, NETSTD, DEFAULT_TEMPLATE, ENSEMBLE_TEMPLATES
from auto_reprogramming.load_model import Load_Reprogramming_Model

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
        '--dataset', choices=["CIFAR10", "CIFAR10-C", "CIFAR100", "ABIDE", "Melanoma", "DR", "SVHN", "GTSRB", "Flowers102", "DTD", "Food101", "EuroSAT", "OxfordIIITPet", "UCF101", "Camelyon17", "Iwildcam", "FMoW", "CLEVR_count"], default="SVHN")
    p.add_argument('--datapath', type=str, default="/home/joytsao/Reprogramming/SVHN")
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--scalibility_rio', type=int, choices=[1, 2, 4, 10, 100], default=1)
    p.add_argument('--scalibility_mode', choices=["equal", "random"], default="equal")  

    args = p.parse_args()

    # set random seed
    set_seed(args.seed)

    # device setting
    device, list_ids = setup_device(1)
    #os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
    device = 'cuda:1'
    print("device: ", device)

    datasets = ["CIFAR10", "CIFAR100","Melanoma", "SVHN", "GTSRB", "Flowers102", "DTD", "Food101", "EuroSAT", "OxfordIIITPet", "UCF101", "FMoW", "CLEVR_count"]
    fig, ax = plt.subplots(len(datasets), 5, figsize=(20,5*len(datasets)))
    for idx, dataset in enumerate(datasets):
        if(dataset == "CLEVR_count"):
            datapath = "/home/joytsao/Reprogramming/CLEVR"
        else:
            datapath = "/home/joytsao/Reprogramming/"+dataset


        ##### Model Setting #####
        channel = 3
        img_resize = IMG_SIZE[dataset]
        class_num = CLASS_NUMBER[dataset]
        # set_train_resize = True
        random_state = args.seed
        ##### End of Setting #####    

        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(1)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(1)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(1)/1024/1024/1024))

        # Load model 
        reprogram_model = Load_Reprogramming_Model(dataset, device, file_path=f"{dataset}_last.pth") #f"{args.dataset}_last.pth" # , set_train_resize=set_train_resize
        # reprogram_model = Load_Reprogramming_Model(args.dataset, device, mapping_method="fully_connected_layer_mapping", pretrained_model="resnet18", mapping=None, scale=1.0)
        if(reprogram_model.no_trainable_resize == 1):
            set_train_resize = False
        else:
            set_train_resize = True
        
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(1)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(1)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(1)/1024/1024/1024))

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
        
        print(reprogram_model.output_mapping.self_definded_map)
        if(set_train_resize == True):
            print(reprogram_model.train_resize.scale)

        wild_ds_list = ["Camelyon17", "Iwildcam", "FMoW"]
        if dataset in wild_ds_list:
            wild_dataset = True
        else:
            wild_dataset = False
                
        CIFAR10_C_mode = None

        # dataloader
        trainloader, testloader, class_names, trainset = DataPrepare(dataset_name=dataset, dataset_dir=datapath, target_size=(
            img_resize, img_resize), mean=NETMEAN[reprogram_model.model_name], std=NETSTD[reprogram_model.model_name], download=False, batch_size=BATCH_SIZE[dataset], random_state=random_state, clip_transform=clip_transform, CIFAR10_C_mode=CIFAR10_C_mode)

        if(args.scalibility_rio != 1):
            trainloader = Data_Scalability(trainset, args.scalibility_rio, BATCH_SIZE[dataset], mode=args.scalibility_mode, random_state=random_state, wild_dataset=wild_dataset) 
        # Plot t_SNE result    

        # Prepare text embedding
        if(reprogram_model.model_name[0:4] == "clip"):
            template_number = 0 # use default template
            reprogram_model.CLIP_Text_Embedding(class_names, template_number) 

        reprogram_model.eval()
        reprogram_model.model.requires_grad_(True)
        net = Network(reprogram_model)

        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(1)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(1)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(1)/1024/1024/1024))


        def swinT_reshape_transform_huggingface(tensor, width, height):
            print("tensor shape:", tensor.shape)
            tensor = tensor[0:-1]
            #print("tensor shape:", tensor.shape)
            #tensor = tensor.permute(0, 2, 3, 1)
            #print("tensor shape:", tensor.shape)
            result = tensor.reshape(1, # tensor.size(0) 
                                    7,
                                    7,
                                    768)  # 128*6 for swin-T # 12 for clip # tensor.size(2) 
            result = result.transpose(2, 3).transpose(1, 2)
            return result
        
        pbar = tqdm(testloader, total=len(testloader), desc=f"Testing", ncols=100)
        mm = nn.Softmax(dim=1)
        #'''
        ii = 0
        for pb in pbar:
            if(wild_dataset == True):
                imgs, labels, _ = pb
            else:
                imgs, labels = pb

            choose_target = ii
            im = torch.unsqueeze(imgs[choose_target], 0)
            logits, purb_img = net(im.to(device))
            logits = mm(logits)
            pred_y = logits.argmax(dim=-1)
            confidence = logits[0, pred_y]
            

            ax[idx][0].imshow(np.transpose(purb_img[0].cpu().detach().numpy(), (1, 2, 0)))
            #ax[idx][0].imshow(np.transpose(np.float32(imgs[choose_target]), (1, 2, 0)))
            ax[idx][0].set_aspect('equal', adjustable='box')
            ax[idx][0].set_title("y=" + class_names[labels[choose_target]] + ", pred_y=" + class_names[pred_y] + "\nconfidence=" + str(confidence.cpu().detach().numpy()))

            input_tensor = torch.unsqueeze(imgs[choose_target], 0) # Create an input tensor image for your model..
            # Note: input_tensor can be a batch tensor with several images!

            reshape_transform = partial(swinT_reshape_transform_huggingface,
                                width=7,
                                height=7) # 7, 7 for swin-T
            #target_layers = [reprogram_model.model.layer1[-1], reprogram_model.model.layer2[-1], reprogram_model.model.layer3[-1], reprogram_model.model.layer4[-1]]#[reprogram_model.model.visual.transformer.resblocks[-1].ln_2] #[reprogram_model.model.layer1[-1]]  #[reprogram_model.model.norm]
            tlid = [0, 4, 8 , -1]
            target_layers = [reprogram_model.model.visual.transformer.resblocks[tlid[0]].ln_2, reprogram_model.model.visual.transformer.resblocks[tlid[1]].ln_2, reprogram_model.model.visual.transformer.resblocks[tlid[2]].ln_2, reprogram_model.model.visual.transformer.resblocks[tlid[3]].ln_2]
            for itx, target_layer in enumerate(target_layers):
                # Construct the CAM object once, and then re-use it on many images:
                cam = GradCAM(model=reprogram_model, target_layers=[target_layer], use_cuda=False, reshape_transform=reshape_transform)#

                # If targets is None, the highest scoring category (for every member in the batch) will be used.
                targets =  [ClassifierOutputTarget(pred_y)]   #[ClassifierOutputTarget(254)] #

                # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

                del cam

                # In this example grayscale_cam has only one image in the batch:
                grayscale_cam = torch.tensor(grayscale_cam[0, :])
                
                heatmap = np.maximum(grayscale_cam, 0)
                # normalize the heatmap
                heatmap /= torch.max(heatmap)

                # draw the heatmap
                img = purb_img[0]
                print(img.shape)
                heatmap =  torch.unsqueeze(torch.unsqueeze(heatmap, 0),0)
                print(heatmap.shape)
                m = nn.Upsample(size=(224,224), mode='bilinear')
                heatmap = m(heatmap)
                print(heatmap.shape)

                im = ax[idx][itx+1].imshow(np.transpose(img.cpu().detach().numpy(), (1, 2, 0)), alpha=0.4)
                ax[idx][itx+1].contourf(heatmap[0,0], alpha=0.4, cmap=plt.cm.jet)
                ax[idx][itx+1].set_aspect('equal', adjustable='box')
                ax[idx][itx+1].set_title("Layer "+str(tlid[itx]))

            break

    plt.savefig("grad-cam-all")