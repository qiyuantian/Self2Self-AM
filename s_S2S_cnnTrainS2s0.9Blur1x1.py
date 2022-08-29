# s_DtiNet_cnnTrain.py
#
#
# QT 2019

# %% load modual
from gc import callbacks
import os
import scipy.io as sio
import numpy as np
import nibabel as nb
import glob
from matplotlib import pyplot as plt
import tensorflow as tf
import nibabel as nib
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import qtlib as qtlib
from s2s_unet1x1 import unet_3d_model
# # Currently, memory growth needs to be the same across GPUs
# %% set up path
dpRoot = os.path.dirname(os.path.abspath('s_S2S_cnnTrainS2s0.9Blur1x1.py'))
os.chdir(dpRoot)

# %% subjects
subjects = sorted(glob.glob(os.path.join(dpRoot, 'mwu*')))

#for ii in np.arange(0, 5):
for ii in np.arange(0, 1):
    sj = os.path.basename(subjects[ii])
    print(sj)
    
    dpSub = os.path.join(dpRoot, sj);        
    
    # %% set up model
    nfilter = 64
    nin = 1
    nout = 1
    
    dtnet = unet_3d_model(nin, nout, filter_num=nfilter)
    dtnet.summary()
    
    adam_opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    dtnet.compile(loss = qtlib.mean_squared_error_weighted, optimizer = adam_opt)
    
    pp = 0.9
    cnnname = 'unet-sw1-blur0.91x1-flip'
    
    dpSave =  os.path.join(dpSub, cnnname) 
    if not os.path.exists(dpSave):
        os.mkdir(dpSave)
        print('create directory')
    
    # %% load data
    # fpImg = os.path.join(dpSub, 't1w-norm', sj + '_wave_norm.nii.gz');
    # fpMask = os.path.join(dpSub, 'wave-bmask-fs', 'brainmask.nii.gz');
    # fpImgBlur = os.path.join(dpSub, 'wave-blur', sj + '_t1w_blurnorm.nii.gz');        
    fpImg = os.path.join(dpSub,'t1w_sim0.3', 't1w_sim_rep1.nii.gz')
    fpMask = os.path.join(dpSub, 't1w','brainmask_fs_dil2.nii.gz')    

    img = nb.load(fpImg).get_data()    
    img = np.expand_dims(img, 3)
    
    # imgblur = nb.load(fpImgBlur).get_data()    
    # imgblur = np.expand_dims(imgblur, 3)
    mask = nb.load(fpMask).get_data()
    if len(mask.shape)==3:
        mask = np.expand_dims(mask, 3)
    else:
        pass
    if len(img.shape)==3:
        img = np.expand_dims(img, 3)
    else:
        pass
    imgnorm, _ = qtlib.normalize_image(img,img,mask)
    imgnorm = imgnorm * mask
    imgblurnorm,kernel = qtlib.blur_image(imgnorm)
    imgblurnorm = imgblurnorm*mask
# # ii,jj,kk=150,125,200
# print(imgblurnorm[ii,jj,kk])
# print("--------")
# print(imgnorm[ii+1,jj,kk])
# print(imgnorm[ii-1,jj,kk])
# print(imgnorm[ii,jj+1,kk])
# print(imgnorm[ii,jj-1,kk])
# print(imgnorm[ii,jj,kk+1])
# print(imgnorm[ii,jj,kk-1])
# numbers = [imgnorm[ii+1,jj,kk],imgnorm[ii-1,jj,kk],imgnorm[ii,jj+1,kk],imgnorm[ii,jj-1,kk],imgnorm[ii,jj,kk+1],imgnorm[ii,jj,kk-1]]
# a=0
# for item in numbers:
#     a = a+item
# print(a/6)
    # %%
    # fpBind = os.path.join(dpSub, 'bind-sw1', sj + '_bind_b96.mat')        
    # bind = sio.loadmat(fpBind)['bind'] - 1
    bind, _ = qtlib.block_ind(mask,sz_block=96)
    idx_start = 0
    idx_step = 17
    DataGen = qtlib.maskimageloader(imgnorm,mask,imgblurnorm,shuffle=True)
    # idx_step = 2
    for jj in np.arange(0, 100):
        DataGen.randommask(pp,bind,idx_step,flip=False)
        DataGen.index_shuffle()        
        # %% training
        nbatch = 1
        
        fnCp = cnnname + '_ep' + np.str(jj)
        fpCp = os.path.join(dpSave, fnCp + '.h5') 
        checkpoint = ModelCheckpoint(fpCp, monitor='val_loss', save_best_only = True)
        
        # history = dtnet.fit(x = [img_block_all, mask_block_all], 
        #                     y = imgres_block_all,
        #                     batch_size = nbatch, 
        #                     validation_split=0.2,
        #                     epochs = 1, 
        #                     callbacks = [checkpoint],
        #                     verbose = 1, 
        #                     shuffle = True) 
        history = dtnet.fit(DataGen.generate_samples("train",1),
                            validation_data = DataGen.generate_samples("valid",1),
                            epochs=1,
                            callbacks = [checkpoint],
                            max_queue_size=10,
                            verbose = 1)

                        
        # save loos
        fpLoss = os.path.join(dpSave, fnCp + '.mat') 
        sio.savemat(fpLoss, {'loss_train':history.history['loss'], 'loss_val':history.history['val_loss']})  
        
        if jj >= 5: # delete intermediate checkout files, which are large
            fnCp = cnnname + '_ep' + np.str(jj - 5)
            fpCp = os.path.join(dpSave, fnCp + '.h5') 
            os.remove(fpCp)

# %% check data
#plt.imshow(img_block_all[5, :, :, 40, 0], clim=(-2., 2.), cmap='gray')
#plt.imshow(imgres_block_all[10, :, :, 40, 0], clim=(0., 1000.), cmap='gray')
#plt.imshow(imgres_block_all[100, :, :, 40, 0] + img_block_all[100, :, :, 40, 0], clim=(0., 1000.), cmap='gray')
#
#plt.imshow(imgres_block_all[100, :, :, 40, 1], clim=(0., 1.), cmap='gray')
#plt.imshow(1. - mask_block_all[100, :, :, 40, 0] + imgres_block_all[100, :, :, 40, 1], clim=(0., 1.), cmap='gray')
#
#
#
#
#tmp = nib.Nifti1Image(img_block_all, np.eye(4))
#nib.save(tmp, 'img_block_all.nii.gz')  
#
#tmp = nib.Nifti1Image(imgres_block_all, np.eye(4))
#nib.save(tmp, 'imgres_block_all.nii.gz')  
#
#tmp = nib.Nifti1Image(mask_block_all, np.eye(4))
#nib.save(tmp, 'mask_block_all.nii.gz')  




