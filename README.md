# DeepDTI Tutorial

![MUnet](https://github.com/qiyuantian/Self2Self-AM/blob/main/imgs/MUnet.png)
**Modified U-Net (MU-Net).** MU-Net is modified from U-Net. All max pooling layers and up-sampling layers are removed. The number of kernels is constant across all layers. The input is a noisy image volume while the output is the residual volume between the input and the high-signal-to-noise ratio (SNR) target image volume (i.e., the noise). Network parameter _k_ = 128 is adopted in this study (~9.3 million parameters).

![MDUnet](https://github.com/qiyuantian/Self2Self-AM/blob/main/imgs/MDUnet.png)
**Modified U-Net with dropout (MDU-Net).** MU-Net is modified from U-Net. The number of kernels is _k_ during the encoding and 2_k_ during the decoding. The input is a noisy image volume with 10% randomly selected voxels replaced by the average of their six neighbouring voxels (i.e., average masking (AM)), while the output is another noisy image volume with the same selected 10% voxels with raw image intensities, on which losses are computed. Network parameter _k_ = 128 is adopted in this study (~5.6 million parameters).

![imgresult](https://github.com/qiyuantian/Self2Self-AM/blob/main/imgs/imgresult.png)
**Image results.** Exemplar image slices of a representative subject from Standard-MPRAGE data (a, i), raw Wave-MPRAGE data (a, ii), and the raw Wave-MPRAGE data denoised by BM4D (a, iii), AONLM (a, iv), supervised denoising using a MU-Net trained on simulation data from HCP subjects (a, v) and the same  MU-Net fine-tuned  on  empirical  data from another subject (a, vi), as well as Self2Self-AM denoising using a MDU-Net trained on simulation data from a HCP subject (c, i), a MDU-Net trained on empirical data from another subject (c, ii), and the same MDU-Net from another subject fine-tuned on the data of the subject for denoising for 1 epochs (c, iii), 5 epochs (c, iv) and 10 epochs (c, v), and a randomly initiated MDU-Net on the data of the subject for denoising for 100 epochs (c, vi), with magnified views of basal ganglia (b, d). The structural similarity index (SSIM) compared to Standard-MPRAGE images and whole-brain averaged vertex-wise gray–white matter boundary sharpness are listed to quantify image quality. The arrowheads highlight the claustrum (magenta) and caudolenticular gray bridges (blue) with fine textures.

![imgresulttable](https://github.com/qiyuantian/Self2Self-AM/blob/main/imgs/imgresulttable.png)
**Image quality quantification.** The group means (± group standard deviations) of the structural similarity index (SSIM) of the raw and denoised Wave-CAIPI images compared to Standard-MPRAGE images as well as the whole-brain averaged vertex-wise gray–white matter boundary sharpness from all 10 subjects. The red color represents the highest metrics while the green color represents the second highest metrics.
## Quick start
install packages in requirements.txt
Run:
python s_S2S_cnnTrainS2s0.9Blur1x1.py
python s_S2S_cnnApplyS2sSw1_0.9Blur1x1.py
## s_S2S_cnnTrainS2s0.9Blur1x1.py

Step-by-step Python tutorial for training the MDUnet in S2S-am.

**Utility functions**

- *s2s_unet1x1.py*: create MDUnet model

- *qtlib.py*: create custom loss functions to only include loss within brain mask, extract blocks from whole brain volume data, normalize brain volume in the brain mask.

**Output**

- *unet-sw1-blur0.91x1-flip/unet-sw1-blur0.91x1-flip_ep99.h5*: MDUnet model trained for 100 epoches

- *unet-sw1-blur0.91x1-flip/unet-sw1-blur0.91x1-flip_ep99.mat*: L2 losses for the training and validation
## s_S2S_cnnTrainS2s1x1.py

Step-by-step Python tutorial for training the MDUnet in S2S.

**Utility functions**

- *s2s_unet1x1.py*: create MDUnet model

- *qtlib.py*: create custom loss functions to only include loss within brain mask, extract blocks from whole brain volume data, normalize brain volume in the brain mask.

**Output**

- *unet-sw11x1/unet-sw11x1_ep99.h5*: MDUnet model trained for 100 epoches

- *unet-sw11x1/unet-sw11x1_ep99.mat*: L2 losses for the training and validation
## s_S2S_cnnApplyS2sSw1_0.9Blur1x1.py

Step-by-step Python tutorial for applying the MDUnet in S2S-am.

**Utility functions**

- *qtlib.py*:  de-normalize brain volume in the brain mask and recover data volume from blocks.

**Output**

- *unet-sw1-blur0.91x1-blur-pred/unet-sw1-blur0.91x1-flip_ep99_img_block_pred0019avg_pad90.nii.gz*: s2s-am denoised result by averaging input with 20 random masks.
## s_S2S_cnnApplyS2sSw1_1x1.py

Step-by-step Python tutorial for applying the MDUnet in S2S-am.

**Utility functions**

- *qtlib.py*:  de-normalize brain volume in the brain mask and recover data volume from blocks.

**Output**

- *unet-sw11x1-pred/unet-sw11x1_ep99_img_block_pred0019avg_pad90.nii.gz*: s2s denoised result by averaging input with 20 random masks.
## **Data availability**

The Wave-MPRAGE and Standard-MPRAGE T1-weighted MRI data of 10 healthy subjects acquired at the MGH Martinos Center will be made freely and publicly available.

## **Refereces**

[1] Tian Q, Li Z, Lo W, Li Z, Bilgic B, Polimeni J, Huang S. Improving the accessibility of deep learning-based denoising for aceelerated brain MRI using self-supervised learning and/or transfer learning.
[2] Tian Q. Improving The Accessibility Of Deep Learning-Based Denoising For MRI Using Transfer Learning And Self-Supervised Learning, Online Power Pitch Oral Presentation. The 2022 Annual Scientific Meeting of ISMRM.

