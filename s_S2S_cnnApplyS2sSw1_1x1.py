# s_cnnApplySep.m
#
#
# (c) Qiyuan Tian, MGH Martinos, 2019 Jan

# %% load moduals
import os
import glob
import scipy.io as sio
import numpy as np
import nibabel as nb
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf
import qtlib as qtlib

# %% set up path
# dpRoot = '/autofs/space/rhapsody_001/users/qtian/S2S-Final'
# os.chdir(dpRoot)
dpRoot = os.path.dirname(os.path.abspath('s_S2S_cnnApplyS2sSw1.py'))
os.chdir(dpRoot)
# %%
subjects = sorted(glob.glob(os.path.join(dpRoot, 'mwu*')))

#for ii in np.arange(1, 20):
# for ii in np.arange(0, 10):
for ii in np.arange(0, 1):   
    sj = os.path.basename(subjects[ii])
    print(sj)
    
    dpSub = os.path.join(dpRoot, sj);     
    dpPred = os.path.join(dpSub, 'unet-sw11x1-pred')
    if not os.path.exists(dpPred):
        os.mkdir(dpPred)
        print('create directory')     
    
    # %% load model 
    fnCnn = 'unet-sw11x1'    
    pp = 0.9
    fnCp = fnCnn + '_ep99'
    
    fpCp = os.path.join(dpSub, fnCnn, fnCp + '.h5') 
    dtnet = 0
    dtnet = load_model(fpCp, custom_objects={'mean_squared_error_weighted': qtlib.mean_squared_error_weighted})

    # fpImg = os.path.join(dpSub, 't1w-norm', sj + '_wave_norm.nii.gz')
    # fpMask = os.path.join(dpSub, 'wave-bmask-fs', 'brainmask.nii.gz')
    # fpImgBlur = os.path.join(dpSub, 'wave-blur', sj + '_t1w_blurnorm.nii.gz')        
    fpImg = os.path.join(dpSub,'t1w_sim0.3', 't1w_sim_rep1.nii.gz')
    fpMask = os.path.join(dpSub, 't1w','brainmask_fs_dil2.nii.gz')

    img = nb.load(fpImg).get_data()    
    img = np.expand_dims(img, 3)

    mask = nb.load(fpMask).get_data()
    mask = np.expand_dims(mask, 3)

    imgnorm,_ = qtlib.normalize_image(img,img,mask)
    # imgblur = nb.load(fpImgBlur).get_data()    
    # imgblur = np.expand_dims(imgblur, 3)
    imgnorm = imgnorm * mask
    imgblurnorm,kernel = qtlib.blur_image(imgnorm)
    imgblurnorm = np.zeros((imgblurnorm*mask).shape)    
    
    # fpBind = os.path.join(dpSub, 'bind-sw1', sj + '_bind_b96.mat')        
    # bind = sio.loadmat(fpBind)['bind'] - 1
    bind, _ = qtlib.block_ind(mask,sz_block=96)
    
    DataGen = qtlib.maskimageloader(imgnorm,mask,imgblurnorm,shuffle=False)
    # for kk in np.arange(0, 20):
    img_preds = []
    count = 0
    for kk in np.arange(0,20):
        DataGen.randommask(pp,bind,idx_step=1)
        DataGen.index_shuffle()   
        
        # img_block_pred = np.zeros(img_block.shape)
        # for mm in np.arange(0, img_block.shape[0]):
            # tmp = dtnet.predict([img_block[mm:mm+1, :, :, :, :], mask_block[mm:mm+1, :, :, :, :]]) 
            # img_block_pred[mm:mm+1, :, :, :, :] = tmp[:, :, :, :, 0:1]
        img_block_pred = dtnet.predict(DataGen.generate_samples('pred',1))
        img_pred,_ = qtlib.block2brain(img_block_pred,bind,mask,pad=3)
        if kk==0:
            img_final = img_pred
        else:
            img_final = img_final + img_pred
        count = count + 1

    img_final = img_final/count
    img_final = qtlib.denormalize_image(img,img_final[:,:,:,0:1],mask)

# %%
    fpPred = os.path.join(dpPred, fnCp + '_img_block_pred' + str(kk).zfill(4) + 'avg_pad90.nii.gz')
    qtlib.save_nii(fpPred,img_final,fpImg)
# %%        