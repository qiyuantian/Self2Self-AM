# qtlib.py
#
# Utiliti functions for training CNN in DeepDTI.
#
# (c) Qiyuan Tian, Harvard, 2021

import numpy as np
import keras.backend as K
from numba import jit
import qtlib
import nibabel as nb
def extract_block(data, inds):
    xsz_block = inds[0, 1] - inds[0, 0] + 1
    ysz_block = inds[0, 3] - inds[0, 2] + 1
    zsz_block = inds[0, 5] - inds[0, 4] + 1
    ch_block = data.shape[-1]
    
    blocks = np.zeros((inds.shape[0], xsz_block, ysz_block, zsz_block, ch_block))
    
    for ii in np.arange(inds.shape[0]):
        inds_this = inds[ii, :]
        blocks[ii, :, :, :, :] = data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :]
    
    return blocks

def mean_squared_error_weighted(y_true, y_pred):
    loss_weights = y_true[:, :, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights # use last channel from grouth-truth data to weight loss from each voxel from first n-1 channels
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights

    return K.mean(K.square(y_pred_weighted - y_true_weighted), axis=-1)

def mean_absolute_error_weighted(y_true, y_pred):
    loss_weights = y_true[:, :, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights
    
    return K.mean(K.abs(y_pred_weighted - y_true_weighted), axis=-1)

def block_ind(mask, sz_block=64, sz_pad=0):

    # find indices of smallest block that covers whole brain
    tmp = np.nonzero(mask);
    xind = tmp[0]
    yind = tmp[1]
    zind = tmp[2]
    
    xmin = np.min(xind); xmax = np.max(xind)
    ymin = np.min(yind); ymax = np.max(yind)
    zmin = np.min(zind); zmax = np.max(zind)
    ind_brain = [xmin, xmax, ymin, ymax, zmin, zmax]; 
    
    # calculate number of blocks along each dimension
    xlen = xmax - xmin + 1
    ylen = ymax - ymin + 1
    zlen = zmax - zmin + 1
    
    nx = int(np.ceil(xlen / sz_block)) + sz_pad
    ny = int(np.ceil(ylen / sz_block)) + sz_pad
    nz = int(np.ceil(zlen / sz_block)) + sz_pad
    
    # determine starting and ending indices of each block
    xstart = xmin
    ystart = ymin
    zstart = zmin
    
    xend = xmax - sz_block + 1
    yend = ymax - sz_block + 1
    zend = zmax - sz_block + 1
    
    xind_block = np.round(np.linspace(xstart, xend, nx))
    yind_block = np.round(np.linspace(ystart, yend, ny))
    zind_block = np.round(np.linspace(zstart, zend, nz))
    
    ind_block = np.zeros([xind_block.shape[0]*yind_block.shape[0]*zind_block.shape[0], 6])
    count = 0
    for ii in np.arange(0, xind_block.shape[0]):
        for jj in np.arange(0, yind_block.shape[0]):
            for kk in np.arange(0, zind_block.shape[0]):
                ind_block[count, :] = np.array([xind_block[ii], xind_block[ii]+sz_block-1, yind_block[jj], yind_block[jj]+sz_block-1, zind_block[kk], zind_block[kk]+sz_block-1])
                count = count + 1
    
    ind_block = ind_block.astype(int)
    
    return ind_block, ind_brain

def normalize_image(imgall, imgresall, mask):
    imgall_norm = np.zeros(imgall.shape)
    imgresall_norm = np.zeros(imgall.shape)
    
    for jj in np.arange(imgall.shape[-1]):
        img = imgall[:, :, :, jj : jj + 1]
        imgres = imgresall[:, :, :, jj : jj + 1]
        
        img_mean = np.mean(img[mask > 0.5])
        img_std = np.std(img[mask > 0.5])
        print("img_mean",img_mean)
        print("img_std",img_std)
        img_norm = (img - img_mean) / img_std * mask;
        imgres_norm = (imgres - img_mean) / img_std * mask;
        
        imgall_norm[:, :, :, jj : jj + 1] = img_norm
        imgresall_norm[:, :, :, jj : jj + 1] = imgres_norm
    return imgall_norm, imgresall_norm
def block2brain(blocks, inds, mask,pad):
    vol_brain = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], blocks.shape[-1]])
    vol_count = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], blocks.shape[-1]])
    xmin = np.min(inds[:,0:2])
    xmax = np.max(inds[:,0:2])
    ymin = np.min(inds[:,2:4])
    ymax = np.max(inds[:,2:4])
    zmin = np.min(inds[:,4:6])
    zmax = np.max(inds[:,4:6])
    bind_flag = []
    bind_flag.append(np.array(inds[:,0:2]==xmin,dtype=float) + np.array(inds[:,0:2]==xmax,dtype=float) )
    bind_flag.append(np.array(inds[:,2:4]==ymin,dtype=float) + np.array(inds[:,2:4]==ymax,dtype=float))
    bind_flag.append(np.array(inds[:,4:6]==zmin,dtype=float) + np.array(inds[:,4:6]==zmax,dtype=float))
    # print("bind_flag--1",bind_flag)
    bind_flag = np.concatenate(bind_flag,-1)
    bind_flag =  bind_flag > 0.5
    bind_flag = 1 - bind_flag
    # print("bind_flag",bind_flag)
    for tt in np.arange(inds.shape[0]):

        inds_this = inds[tt, :]
        flag = bind_flag[tt, :]
        tmp1 = blocks[tt,:,:,:,:]
        mask1 = np.zeros(tmp1.shape)
        ind_tmp1 = [0,96,0,96,0,96]
        ind_tmp2 = [3,93,3,93,3,93]
        ind_tmp = ind_tmp1*(1-flag)+ind_tmp2*flag
        # print("ind_tmp",ind_tmp)
        mask1[ind_tmp[0]:ind_tmp[1],ind_tmp[2]:ind_tmp[3],ind_tmp[4]:ind_tmp[5]] = 1
        tmp1 = tmp1*mask1
        # print("1",vol_brain[inds_this[0]+pad:inds_this[1]-pad+1, inds_this[2]+pad:inds_this[3]-pad+1, inds_this[4]+pad:inds_this[5]-pad+1, :].shape)
        # print("2",blocks[tt, pad:-pad+1, pad:-pad+1, pad:-pad+1, :].shape)
        # vol_brain[inds_this[0]+pad:inds_this[1]-pad+1, inds_this[2]+pad:inds_this[3]-pad+1, inds_this[4]+pad:inds_this[5]-pad+1, :] = \
        #         vol_brain[inds_this[0]+pad:inds_this[1]-pad+1, inds_this[2]+pad:inds_this[3]-pad+1, inds_this[4]+pad:inds_this[5]-pad+1, :] + blocks[tt, pad:-pad, pad:-pad, pad:-pad, :]
        
        # vol_count[inds_this[0]+pad:inds_this[1]-pad+1, inds_this[2]+pad:inds_this[3]-pad+1, inds_this[4]+pad:inds_this[5]-pad+1, :] = \
        #         vol_count[inds_this[0]+pad:inds_this[1]-pad+1, inds_this[2]+pad:inds_this[3]-pad+1, inds_this[4]+pad:inds_this[5]-pad+1, :] + 1.
        vol_brain[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] = \
                vol_brain[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] + tmp1
        
        vol_count[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] = \
                vol_count[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] + mask1  
    vol_count[vol_count < 0.5] = 1.
    vol_brain = vol_brain / vol_count 
    
    vol_brain = vol_brain * mask
    vol_count = vol_count * mask
    
    return vol_brain, vol_count 
def blur_image(img):
    kernel = np.zeros([3,3,3])
    kernel[0,1,1] = 1
    kernel[1,0,1] = 1
    kernel[1,1,0] = 1
    kernel[1,1,2] = 1
    kernel[1,2,1] = 1
    kernel[2,1,1] = 1
    kernel = kernel/np.sum(kernel)
    avgmask = np.zeros(img.shape)
    for jj in range(0,img.shape[-1]):
        avgmask[:,:,:,jj:jj+1] = conv3D(img[:,:,:,jj:jj+1],kernel)
    return avgmask,kernel
def save_nii(fpNii, data, fpRef):
    new_header = header=nb.load(fpRef).header.copy()    
    new_img = nb.nifti1.Nifti1Image(data, None, header=new_header)    
    nb.save(new_img, fpNii)  
@jit(cache=True)
def conv3D(img,kernel):
    convresult = np.zeros(img.shape)
    for ii in range(1,img.shape[0]-1):
        for jj in range(1,img.shape[1]-1):
            for kk in range(1,img.shape[2]-1):
                convresult[ii,jj,kk,0] = np.sum(img[ii-1:ii+2,jj-1:jj+2,kk-1:kk+2,0]*kernel)
    return convresult

class maskimageloader:
    def __init__(self,imgnorm,mask,imgblurnorm,shuffle):
        self.mask = mask
        self.imgnorm = imgnorm
        self.imgblurnorm = imgblurnorm
        self.shuffle = shuffle
    def randommask(self,pp,bind,idx_step,flip = False):
        self.img_block_all = np.zeros(1)
        self.imgres_block_all = np.zeros(1)
        self.mask_block_all = np.zeros(1)
        for kk in range(0,idx_step):
            print(kk)
            rmask = np.random.binomial(1,pp,self.mask.shape)
            imgmasked = self.imgnorm * rmask * self.mask + self.imgblurnorm * (1. - rmask) * self.mask
            lmask = (1. - rmask) * self.mask
            imgres_masked = self.imgnorm * lmask
            img_block = qtlib.extract_block(imgmasked, bind)
            imgres_block = qtlib.extract_block(imgres_masked, bind)
            mask_block = qtlib.extract_block(lmask, bind)
            
            imgres_block = np.concatenate((imgres_block, mask_block), axis=-1)
            
            if self.imgres_block_all.any():
                self.img_block_all = np.concatenate((self.img_block_all, img_block), axis=0)
                self.imgres_block_all = np.concatenate((self.imgres_block_all, imgres_block), axis=0)
                self.mask_block_all = np.concatenate((self.mask_block_all, mask_block), axis=0)
            else:
                self.img_block_all = img_block
                self.imgres_block_all = imgres_block
                self.mask_block_all = mask_block 
        if flip:
            self.img_block_all = np.concatenate([self.img_block_all,np.flip(self.img_block_all,1)],0)   
            self.imgres_block_all = np.concatenate([self.imgres_block_all,np.flip(self.imgres_block_all,1)],0)   
            self.mask_block_all = np.concatenate([self.mask_block_all,np.flip(self.mask_block_all,1)],0)   
    def generate_samples(self,mode,nbatch):
        index = self.index
        train_ratio = 0.8
        train_valid_gap = int(self.img_block_all.shape[0]*train_ratio)
        if mode=="train":
            for count in range(0,train_valid_gap,nbatch):
            # for count in range(0,1):
                if index[count]<=self.img_block_all.shape[0]:
                    img_block = self.img_block_all[index[count]:index[count]+nbatch,:,:,:,:]
                    imgres_block = self.imgres_block_all[index[count]:index[count]+nbatch,:,:,:,:]
                    mask_block = self.mask_block_all[index[count]:index[count]+nbatch,:,:,:,:]
                else:
                    img_block = np.flip(self.img_block_all[index[count]:index[count]+nbatch,:,:,:,:],1)
                    imgres_block = np.flip(self.imgres_block_all[index[count]:index[count]+nbatch,:,:,:,:],1)
                    mask_block = np.flip(self.mask_block_all[index[count]:index[count]+nbatch,:,:,:,:],1)
                x = [img_block,mask_block]
                y = imgres_block
                yield (x,y)
        elif mode=="valid":
            for count in range(train_valid_gap,self.img_block_all.shape[0],nbatch):
            # for count in range(0,1):
                if index[count]<=self.img_block_all.shape[0]:
                    img_block = self.img_block_all[index[count]:index[count]+nbatch,:,:,:,:]
                    imgres_block = self.imgres_block_all[index[count]:index[count]+nbatch,:,:,:,:]
                    mask_block = self.mask_block_all[index[count]:index[count]+nbatch,:,:,:,:]
                else:
                    img_block = np.flip(self.img_block_all[index[count]:index[count]+nbatch,:,:,:,:],1)
                    imgres_block = np.flip(self.imgres_block_all[index[count]:index[count]+nbatch,:,:,:,:],1)
                    mask_block = np.flip(self.mask_block_all[index[count]:index[count]+nbatch,:,:,:,:],1)
                x = [img_block,mask_block]
                y = imgres_block
                yield (x,y)
        elif mode=="pred":
            for count in range(0,self.img_block_all.shape[0],nbatch):
            # for count in range(0,1):
                img_block = self.img_block_all[index[count]:index[count]+nbatch,:,:,:,:]
                imgres_block = self.imgres_block_all[index[count]:index[count]+nbatch,:,:,:,:]
                mask_block = self.mask_block_all[index[count]:index[count]+nbatch,:,:,:,:]
                x = [img_block,mask_block]
                y = imgres_block
                yield (x,y)
        else:
            assert(0)
    def index_shuffle(self):
        self.index = np.arange(self.img_block_all.shape[0])
        if self.shuffle:
            np.random.shuffle(self.index)
        else:
            pass

def denormalize_image(imgall, imgnormall, mask):
    imgresall_denorm = np.zeros(imgnormall.shape)
    
    for jj in np.arange(imgall.shape[-1]):
        img = imgall[:, :, :, jj : jj + 1]
        imgres = imgnormall[:, :, :, jj : jj + 1]
        
        img_mean = np.mean(img[mask > 0.5])
        img_std = np.std(img[mask > 0.5])
        print("img_mean",img_mean)
        print("img_std",img_std)
        imgres_norm = (imgres * img_std + img_mean) 
        
        imgresall_denorm[:, :, :, jj : jj + 1] = imgres_norm
    return imgresall_denorm