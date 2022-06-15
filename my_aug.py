# coding=utf-8
"""for attention data aug"""
import sys,os
import numpy as np 
import random
from PIL import Image
from PIL import ImageEnhance
from scipy.ndimage import filters
import cv2
import cfgs


def resize_pil_img(img_pil):
    img_3channel = img_pil.convert('RGB')
    img_arr = np.array(img_3channel)
    img_pad = resize_padding(img_arr)
   
    return Image.fromarray(np.uint8(img_pad)).convert('L')

def resize_padding(img, strip=True):
    
    IMAGE_HEIGHT = cfgs.IMAGE_H
    IMAGE_WIDTH = cfgs.IMAGE_W
    
    h,w,d = img.shape
    
    if strip:
        new_img = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,3))
        new_img = add_stripe(new_img)
    else:
        new_img = np.ones((IMAGE_HEIGHT,IMAGE_WIDTH,3))*127

    if w*1.0/h > (IMAGE_WIDTH*1.0/IMAGE_HEIGHT): # w长
        h_hat = int(IMAGE_WIDTH*h*1.0/w)
        img = cv2.resize(img, (IMAGE_WIDTH, h_hat), interpolation=cv2.INTER_AREA)
        r0 = int((IMAGE_HEIGHT-h_hat)/2)
        r1 = r0+h_hat
        new_img[r0:r1,:,:] = img
    else: # h高
        w_hat = int(IMAGE_HEIGHT*w*1.0/h)
        img = cv2.resize(img, (w_hat, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        c0 = int((IMAGE_WIDTH-w_hat)/2)
        c1 = c0+w_hat
  
        new_img[:,c0:c1,:]=img
    return new_img

def add_stripe(img_in):

    h, w, _ = img_in.shape

    num = h // 4

    for ind in range(num):
        if ind % 2 ==0:
            img_in[ind*4:ind*4+4,:,:] += 255

    return img_in  

def aug_bright(img_pil):
    enh_bri = ImageEnhance.Brightness(img_pil)
    bri_list = [0.5,1.5,2]
    brightness = random.choice(bri_list)

    img_aug = enh_bri.enhance(brightness)
    return img_aug

def aug_blur(img_pil,depth):
    # 随机高斯模糊
    delta_list = [1,1.5,2,2.5]
    img_data = np.array(img_pil)
    delta = random.choice(delta_list)
    if depth ==1:
        img_data = filters.gaussian_filter(img_data,delta)
    else:
        for i in range(3):
            img_data[:,:,i] = filters.gaussian_filter(img_data[:,:,i],delta)
    img_pil = Image.fromarray(np.uint8(img_data))

    return img_pil

def aug_camera(img_pil,degree=8, angle=60):
    # 模拟运动模糊（相机抖动）
    degree = random.randint(3, 10)
    angle = random.randint(0, 360)
    image = np.array(img_pil)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
 
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
 
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    img_pil = Image.fromarray(np.uint8(blurred))

    return img_pil

def aug_mixup(img_now,img_next):
    lamda = 0.8
    # 模拟透字现象，对图片做水平翻转
    img_flip = img_next.transpose(Image.FLIP_LEFT_RIGHT)
    arr_now = np.array(img_now)
    arr_flip = np.array(img_flip)
    mix_w = max(arr_now.shape[1], arr_flip.shape[1])
    mix_h = max(arr_now.shape[0], arr_flip.shape[0])

    # 初始化2个空白矩阵，尺寸一致
    im1 = np.zeros((mix_h,mix_w))
    im2 = np.zeros((mix_h,mix_w))

    # 赋值
    im1[:arr_now.shape[0],:arr_now.shape[1]] = arr_now
    im2[:arr_flip.shape[0],:arr_flip.shape[1]] = arr_flip
    # 做mixup
    arr_mix = im1*lamda+(1-lamda)*im2

    return Image.fromarray(np.uint8(arr_mix))

def bgr_mixup(img_now):
    IMAGE_HEIGHT = cfgs.IMAGE_H
    IMAGE_WIDTH = cfgs.IMAGE_W
 
    threshold = 170
    
    bg_dir = './datasets/backgrd/'
    bg_list = [os.path.join(bg_dir,f) for f in os.listdir(bg_dir)]
    bg_path = random.choice(bg_list)
   
    # bg_data = np.array(Image.open(bg_path).convert('L'))

    bg_data = Image.open(bg_path).convert('L').resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    bg_data = np.array(bg_data)

    mask = np.where(bg_data>threshold)
    nomask = np.where(bg_data<=threshold)
    bg_data[mask] = 255
    k = 255.0/threshold 
    bg_data[nomask] = bg_data[nomask]*k
    bg_data = np.uint8(bg_data)
    # imgcat(bg_data)
    arr_now = np.array(img_now)

    h, w = arr_now.shape
    resize_now = img_now.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    arr_resize = np.array(resize_now)

    bg = bg_data/255.
    now = arr_resize/255

    arr_mix = bg * now
    arr_mix = np.round((arr_mix * 255. )).astype(np.uint8)
    # imgcat(arr_mix)

    mix_pil = Image.fromarray(arr_mix)
    mix_pil = mix_pil.resize((w, h))

    return mix_pil

def aug_random(img_now,img_next,depth):
    
    # 0.5的概率选择要不要做数据增强
    index = np.random.randint(0,10)
    aug_type = random.choice(['blur', 'camera', 'mixup', 'bright', 'mixup2'])
   
    if index>5:
        return img_now
    else:
        if aug_type=='blur':
            img_aug = aug_blur(img_now,depth)

        if aug_type=='camera':
            img_aug = aug_camera(img_now,depth)
        
        if aug_type=='mixup':
            img_aug = aug_mixup(img_now,img_next)
        
        if aug_type=='bright':
            img_aug = aug_bright(img_now)

        if aug_type=='mixup2':
            img_aug = bgr_mixup(img_now)
        return img_aug


def doEageAug(img_array):

    h, w = img_array.shape

    exs_w = [np.random.randint(0, w // 10 + 1) for i in range(4)]
    exs_h = [np.random.randint(0, h // 10 + 1) for i in range(4)]

    # 调整位置
    x1, y1 = exs_w[0], exs_h[0]
    x2, y2 = w - 1 - exs_w[1], exs_h[1]
    x3, y3 = exs_w[2], h - 1 - exs_h[2]
    x4, y4 = w - 1 - exs_w[3], h - 1 - exs_h[3]

    src = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.float32)
    dst = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], np.float32)

    p = cv2.getPerspectiveTransform(src, dst)

    img_ex = cv2.warpPerspective(img_array, p, (w, h), borderValue=127)
    return img_ex


if __name__ == '__main__':
    image_p1 = './check/0002_29.jpg'
    # img_pil = Image.open(image_p1).convert('L')

    img_3channel = Image.open(image_p1)
    img_arr = np.array(img_3channel)
    img_pad = resize_padding(img_arr)
    img_pad = np.uint8(img_pad)
    imgcat(img_pad)

    # out = resize_pil_img(imp_pil1)

    # image_p2 = './images/0001_6.jpg'
    # imp_pil2 = Image.open(image_p2).convert('L')
  
    # imp_pil = aug_bright(imp_pil1)
   
    # imp_pil.save('./bright.png')


