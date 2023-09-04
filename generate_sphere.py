import numpy as np
from skimage.io import imsave
import os

img = np.zeros((100,100,100))

center = (50,50,50)
center = np.array(center)

ray = 50

def gen_sphere(img, center, ray):
    x,y,z = np.meshgrid(*[np.arange(d) for d in img.shape])
    x,y,z = x-center[0],y-center[1],z-center[2]
    img[np.sqrt(x*x+y*y+z*z)<ray]=1
    return img

def alea_sphere(img_size, range_ray, range_center):
    img_size = np.array(img_size)
    range_ray = np.array(range_ray)
    range_center = np.array(range_center)
    ray = np.random.randint(range_ray[0], range_ray[1])
    center = np.array([np.random.randint(rc[0], rc[1]) for rc in range_center])
    center = np.where(np.greater(center + ray,img_size), img_size-ray, center)
    center = np.where(np.less(center-ray, 0), ray, center)
    img = np.zeros(img_size)
    return gen_sphere(img, center, ray), 4*np.pi*(ray**3)/3

def sphere_dir(path, n=20):
    for i in range(n):
        msk, vol = alea_sphere((100,100,100),(10,30),((20,80),(20,80),(20,80)))
        imsave(os.path.join(path, str(vol)+'.tif'), msk)
        print("Measured volume:", np.sum(msk))
        print("Actual volume:", vol)

sphere_dir("img\\")
# msk = gen_sphere(img, center, ray)
# msk, vol = alea_sphere((100,100,100),(10,30),((20,80),(20,80),(20,80)))
# imsave("tmp.tif", msk)
# print(len(np.meshgrid(np.arange(10), np.arange(10), np.arange(10))))
