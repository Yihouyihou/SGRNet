# -*- coding: utf-8 -*-


from net.UNet_pytorch import UNet
import matplotlib
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from scipy import io
from dataset import MyDataSet
from PIL import Image
x_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

y_transforms = transforms.ToTensor()


for i in range(101,105):
    img=Image.open('C:/Users/zya/Desktop/ACCESS/BCFM/定位（figure7）/'+str(i)+'cell.png')
    img=np.array(img)
    np.save('C:/Users/zya/Desktop/ACCESS/BCFM/定位（figure7）/'+str(i)+'.npy',img)
    print(img.dtype)
print('done')

start=101

end=105

patch_size=256

img_size=256

# test_dataset = MyDataSet(root='C:/Users/zya/Desktop/UW/dataset/', start=start, end=end, label=None,
#                          transform=x_transforms, target_transform=y_transforms)
# test_dataset = MyDataSet(root='C:/Users/zya/Desktop/PD-L1-1/100_img_patch/', start=start, end=end, label=None,
#                           transform=x_transforms, target_transform=y_transforms)
test_dataset = MyDataSet(root='C:/Users/zya/Desktop/BCFM/dataset_test/', start=0,end=1600, label=None,
                         transform=x_transforms, target_transform=y_transforms)
test_dataset = MyDataSet(root='C:/Users/zya/Desktop/ACCESS/BCFM/定位（figure7）/', start=101,end=105, label=None,
                         transform=x_transforms, target_transform=y_transforms)
# ----------------------------- Evaluating model------------------------------


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

model = UNet(3, 1).to(device)

model_name='pretrain_PDL1'
#checkpoint = torch.load('C:/Users/zya/Desktop/train0424_1641/train0424_1641.pth.tar')
#checkpoint = torch.load('./'+model_name+'.pth.tar')
checkpoint = torch.load('C:/Users/zya/Desktop/BCFM/SMRRN_BCFM_2.601.pth.tar')
#checkpoint = torch.load('./pretrain1731.pth.tar')

model.load_state_dict(checkpoint['state_dict'])
#model.load_state_dict(torch.load('C:/Users/zya/Desktop/PSU/save_model/model_93_1.pth'))

#print('alfa:',checkpoint['alfa'])
#print('lamda:',checkpoint['lamda'])
# print(model.state_dict())
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']


model.eval()

pred_label = np.zeros((end-start, patch_size, patch_size))
#pred_label = np.zeros((4, 452, 612))

test_dataloader = DataLoader(test_dataset, batch_size=1)

for i, test_data in tqdm(enumerate(test_dataloader, 0), total=end-start, ncols=50):
    imgs = test_data
    imgs = imgs.float()
    # maps = maps.float()
    imgs = imgs.cuda()
    # maps = maps.cuda()

    output = model(imgs)

    p = output.cpu()
    p = p.detach().numpy()
    #print(p.shape)
    #np.save('C:/Users/zya/Desktop/PD-L1-1/mask_pred_set/'+str(i+3400)+'_pred_label.npy',p)
    #print('保存第'+str(i)+'个成功')
    pred_label[i, :, :] = p
    matplotlib.image.imsave('C:/Users/zya/Desktop/ACCESS/BCFM/'+str(i+100)+'_pred_label.png',p[0,0,:,:],cmap='jet')


# print('VAL LOSS: {:.6f}'.format(mse / (i + 1)))


def count_cell1(density_patch, patch_size, size,img_size):
    stride_h = patch_size
    stride_h = 452
    stride_w = patch_size
    stride_w = 612

    patch_size_h = patch_size
    patch_size_h = 452
    patch_size_w = patch_size
    patch_size_w = 612

    h = img_size
    h = 452
    w = img_size
    w = 612
    D = np.zeros((h, w, size))
    t = 0
    for index in range(size):
        for i in range(0, h, stride_h):
            for j in range(0, w, stride_w):
                d = density_patch[t, :, :]
                D[i:i + patch_size_h, j:j + patch_size_w, index] += d
                t = t + 1
        #matplotlib.image.imsave('C:/Users/zya/Desktop/PD-L1-2/test/save/save_2/' + str(index+1) + '.png', D[:,:,index], cmap='jet')
    # D = D/4
    count = sum(sum(D)) / 1000

    return D, count


def count_cell(density_patch, patch_size, size,img_size):
    stride_h = patch_size
    stride_w = patch_size

    patch_size_h = patch_size
    patch_size_w = patch_size

    h = img_size
    w = img_size
    D = np.zeros((h, w, size))
    t = 0
    for index in range(size):
        for i in range(0, h, stride_h):
            for j in range(0, w, stride_w):
                d = density_patch[t, :, :]
                D[i:i + patch_size_h, j:j + patch_size_w, index] += d
                t = t + 1
        #matplotlib.image.imsave('C:/Users/zya/Desktop/PD-L1-2/test/save/save_2/' + str(index+1) + '.png', D[:,:,index], cmap='jet')
    # D = D/4
    count = sum(sum(D)) / 1000

    return D, count

size = (end-start)//((img_size//patch_size)**2)

gt=np.array([196,284,68,322,168,70,72,26,195,187,43,62,130,186,86,95,156,96,121,311,389,262,205,375])
gt = np.load('C:/Users/zya/Desktop/PD-L1-1/gt.npy')
gt = np.load('C:/Users/zya/Desktop/UW/val_gt.npy')
gt = np.load('C:/Users/zya/Desktop/BCFM/val_label.npy')
print(gt)
val_map, val_count = count_cell(pred_label, patch_size, size,img_size)
#val_map, val_count = count_cell(pred_label, patch_size, 3,img_size)
print(val_map.shape)
# for i in range(size):
#     matplotlib.image.imsave('C:/Users/zya/Desktop/BCFM/save/save_1/'+str(i)+'.png',val_map[:,:,i],cmap='jet')
#np.save('C:/Users/zya/Desktop/train0424_1641/map_pred.npy',val_map)
#print(val_count)
#print('MAE:',sum(abs(gt-val_count))/size)
# np.savetxt('./UW.csv', val_count, delimiter=',')
io.savemat('C:/Users/zya/Desktop/ACCESS/BCFM/定位（figure7）/BCFM.mat',{'map_pred':val_map})