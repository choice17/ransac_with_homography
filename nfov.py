# Copyright 2017 Nitish Mutha (nitishmutha.com)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import pi
import numpy as np
from PIL import Image
import cv2
import time

class NFOV():
    def __init__(self, height=400, width=400, v2=0, fov=[0.6,0.6], eqWidth=pi/2):
        self.FOV = fov
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0
        self.height = height
        self.width = width
        self.v2 = v2
        self.eqWidth = eqWidth
        if self.v2:
            self.screen_points = self._get_screen_img_v2()
        else:
            self.screen_points = self._get_screen_img()

    def _get_coord_rad(self, isCenterPt, center_point=None):
        a = (center_point * 2 - 1) * np.array([self.PI, self.PI_2]) \
            if isCenterPt \
            else \
            (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (
                np.ones(self.screen_points.shape) * self.FOV)
        print(self.FOV)
        return a

    def _get_screen_img(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        return np.array([xx.ravel(), yy.ravel()]).T

    def _get_screen_img_v2(self):
        xx, yy = np.meshgrid(np.linspace(0, 1, self.width), np.linspace(0, 1, self.height))
        return np.dstack([xx,yy])

    def _calcSphericaltoGnomonic(self, convertedScreenCoord):
        x = convertedScreenCoord.T[0]
        y = convertedScreenCoord.T[1]

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5

        return np.array([lon, lat]).T
    
    def _calcSphericaltoGnomonic_v2(self, convertedScreenCoord):
        x = convertedScreenCoord[:,:,0]
        y = convertedScreenCoord[:,:,1]

        rou = np.sqrt(x ** 2 + y ** 2)
        c = np.arctan(rou)
        sin_c = np.sin(c)
        cos_c = np.cos(c)

        lat = np.arcsin(cos_c * np.sin(self.cp[1]) + (y * sin_c * np.cos(self.cp[1])) / rou)
        lon = self.cp[0] + np.arctan2(x * sin_c, rou * np.cos(self.cp[1]) * cos_c - y * np.sin(self.cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.eqWidth  + 1.) * 0.5

        return np.dstack([lon, lat])

    def _calcSphericaltoGnomonic_v3(self):
        x = self.x
        y = self.y

        rou = self.rou
        c = self.c
        sin_c = self.sin_c
        cos_c = self.cos_c
        cp = self.cp
        lat = np.arcsin(cos_c * np.sin(cp[1]) + (y * sin_c * np.cos(cp[1])) / rou)
        lon = cp[0] + np.arctan2(x * sin_c, rou * np.cos(cp[1]) * cos_c - y * np.sin(cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.eqWidth + 1.) * 0.5

        return np.dstack([lon, lat])

    def _bilinear_interpolation(self, screen_coord):
        uf = np.mod(screen_coord.T[0],1) * self.frame_width  # long - width
        vf = np.mod(screen_coord.T[1],1) * self.frame_height  # lat - height

        x0 = np.floor(uf).astype(int)  # coord of pixel to bottom left
        y0 = np.floor(vf).astype(int)
        x2 = np.add(x0, np.ones(uf.shape).astype(int))  # coords of pixel to top right
        y2 = np.add(y0, np.ones(vf.shape).astype(int))

        base_y0 = np.multiply(y0, self.frame_width)
        base_y2 = np.multiply(y2, self.frame_width)

        A_idx = np.add(base_y0, x0)
        B_idx = np.add(base_y2, x0)
        C_idx = np.add(base_y0, x2)
        D_idx = np.add(base_y2, x2)

        flat_img = np.reshape(self.frame, [-1, self.frame_channel])

        A = np.take(flat_img, A_idx, axis=0)
        B = np.take(flat_img, B_idx, axis=0)
        C = np.take(flat_img, C_idx, axis=0)
        D = np.take(flat_img, D_idx, axis=0)

        wa = np.multiply(x2 - uf, y2 - vf)
        wb = np.multiply(x2 - uf, vf - y0)
        wc = np.multiply(uf - x0, y2 - vf)
        wd = np.multiply(uf - x0, vf - y0)

        # interpolate
        AA = np.multiply(A, np.array([wa, wa, wa]).T)
        BB = np.multiply(B, np.array([wb, wb, wb]).T)
        CC = np.multiply(C, np.array([wc, wc, wc]).T)
        DD = np.multiply(D, np.array([wd, wd, wd]).T)
        nfov = np.reshape(np.round(AA + BB + CC + DD).astype(np.uint8), [self.height, self.width, 3])
        import matplotlib.pyplot as plt
        plt.imshow(nfov)
        plt.show()
        return nfov

    def _bilinear_interpolation_v2(self, screen_coord):
        self.frame[0,0,:] = 0
        uf = screen_coord[:,:,0] * self.frame_width # long - width
        vf = screen_coord[:,:,1] * self.frame_height # lat - height

        #mask = (uf < 0) | (uf >= (self.frame_width - 2)) | (vf < 0) | (vf > (self.frame_height - 2))
        #uf[mask] = 0
        #vf[mask] = 0
        mask = uf < 0
        uf[mask] = self.frame_width + uf[mask]
        mask = uf >= (self.frame_width - 2)
        uf[mask] -= (self.frame_width - 2)
        mask = vf < 0
        vf[mask] = self.frame_height + vf[mask]
        mask = vf >= (self.frame_height - 2)
        vf[mask] -= (self.frame_height - 2)


        x0 = uf.astype(np.int32)  # coord of pixel to bottom left
        y0 = vf.astype(np.int32)
        
        wx = uf - x0
        wy = vf - y0

        nfov = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        for i in range(3):
            l0 = self.frame[y0, x0, i] * ( 1 - wx ) +  self.frame[y0, x0 + 1, i] * wx
            l1 = self.frame[y0+1, x0, i] * ( 1 - wx ) +  self.frame[y0, x0 + 1, i] * wx
            nfov[:,:,i] = l0 * (1 - wy) + l1 * wy

        #import matplotlib.pyplot as plt
        #plt.imshow(nfov)
        #plt.show()
        return nfov

    def toNFOV(self, frame, center_point):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]

        self.cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)
        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        spericalCoord = self._calcSphericaltoGnomonic(convertedScreenCoord)
        return self._bilinear_interpolation(spericalCoord)

    def toNFOV2(self, frame, center_point):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]

        self.cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)
        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        spericalCoord = self._calcSphericaltoGnomonic_v2(convertedScreenCoord)
        return self._bilinear_interpolation_v2(spericalCoord)

    def prepare(self, frame):
        self.frame = frame
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.frame_channel = frame.shape[2]
        convertedScreenCoord = self._get_coord_rad(isCenterPt=False)
        self.coord = np.zeros(convertedScreenCoord.shape, dtype=np.int32)
        self.x = convertedScreenCoord[:,:,0]
        self.y = convertedScreenCoord[:,:,1]
        self.rou = np.sqrt(self.x ** 2 + self.y ** 2)
        self.c = np.arctan(self.rou)
        self.sin_c = np.sin(self.c)
        self.cos_c = np.cos(self.c)

    def toNFOV3(self, center_point):
        self.cp = self._get_coord_rad(center_point=center_point, isCenterPt=True)
        spericalCoord = self._calcSphericaltoGnomonic_v3()
        return self._bilinear_interpolation_v2(spericalCoord)


class GnomonicProjection(object):
    def __init__(self, fov=[0.2,0.2], dst_w=600, dst_h=400):
        self.width = dst_w
        self.height = dst_h
        self.src = None
        self.src_h  = self.src_w = None
        self.fov = fov
        self.pi2 = pi * 2
        self.setFov(fov)

    def dstCoordGen(self, dst_w, dst_h, fov):
        fov = np.tan([i/2 for i in fov])
        lin_x = np.linspace(-fov[1], fov[1], dst_w)
        lin_y = np.linspace(-fov[0], fov[0], dst_h)
        fovCoord = np.dstack(np.meshgrid(lin_x, -lin_y))
        return fovCoord

    def getRotateMat(latn, longn):
        rx = np.array([[1,0,0],
                       [0,np.cos(longn),-np.sin(longn)],
                       [0,np.sin(longn),np.cos(longn)]])
        ry = np.array([[np.cos(latn),0,np.sin(latn)],
                       [0,1,0],
                       [-np.sin(latn),0,np.cos(latn)]])
        rot = np.dot(ry, rx).T
        return rot

    def rotateCoord(fovCoord, rotMat):
        # coord [HxWx3] -> rMat [3x3]
        coord = fovCoord.dot(rotMat)
        return coord

    def remap(self, coord, dst_h, dst_w, src_h, src_w):
        # remap sphere
        u = np.arctan2(coord[...,0], coord[...,2])
        p = np.sqrt(coord[...,0] * coord[...,0] + coord[...,2] * coord[...,2])
        v = np.arctan2(coord[...,1], p)

        # remap equilrectangle
        x = (u / self.pi2 + 0.5) * src_w - 0.5
        y = (-v / np.pi + 0.5) * src_h - 0.5
        return x, y
        #return x.reshape(dst_h, dst_w), y.reshape(dst_h, dst_w)
        
    def wrapCoord(self, src, coord_x, coord_y):
        src_h, src_w, _ = src.shape
        uf = coord_x
        vf = coord_y

        mask = uf < 0
        uf[mask] = src_w + uf[mask]
        mask = uf >= (src_w - 2)
        uf[mask] -= (src_w - 2)
        mask = vf < 0
        vf[mask] = src_h + vf[mask]
        mask = vf >= (src_h - 2)
        vf[mask] -= (src_h - 2)

        """
        bilinear interpolation
        """
        x0 = uf.astype(np.int32)  # coord of pixel to bottom left
        y0 = vf.astype(np.int32)
        
        wx = uf - x0
        wy = vf - y0

        nfov = np.zeros((self.height,self.width, 3), dtype=np.uint8)
        
        for i in range(3):
            l0 = src[y0, x0, i] * ( 1 - wx ) +  src[y0, x0 + 1, i] * wx
            l1 = src[y0+1, x0, i] * ( 1 - wx ) +  src[y0, x0 + 1, i] * wx
            nfov[:,:,i] = l0 * (1 - wy) + l1 * wy

        return nfov

    def prepare(self, frame):
        self.src = frame
        self.src_h, self.src_w, _ = self.src.shape

    def setFov(self, fov):
        self.fovCoord = np.ones((self.height,self.width,3))
        self.fovCoord[:,:,:2] = self.dstCoordGen(self.width, self.height, self.fov)

    def project(self, center_point):
        longn, latn = center_point

        rotMat = GnomonicProjection.getRotateMat(longn,latn)
        coord = self.fovCoord.dot(rotMat)
        x, y = self.remap(coord, self.height, self.width, self.src_h, self.src_w)
  
        return self.wrapCoord(self.src, x, y)

    def remap(self, coord, dst_h, dst_w, src_h, src_w):
        # remap sphere
        u = np.arctan2(coord[...,0], coord[...,2])
        p = np.sqrt(coord[...,0] * coord[...,0] + coord[...,2] * coord[...,2])
        v = np.arctan2(coord[...,1], p)

        # remap equilrectangle
        x = (u / self.pi2 + 0.5) * src_w - 0.5
        y = (-v / np.pi + 0.5) * src_h - 0.5
        return x, y

    def projectv2(self, center_point):
        longn, latn = center_point
        x, y = eq_to_pers(self.src, self.fov[0], longn, latn, self.height, self.width)
        return self.wrapCoord(self.src, x[...,0], y[...,0])

def gen_xyz(fov, u, v, out_h, out_w):
    out = np.ones((out_h, out_w, 3), np.float32)

    x_rng = np.linspace(-np.tan(fov / 2), np.tan(fov / 2), num=out_w, dtype=np.float32)
    y_rng = np.linspace(-np.tan(fov / 2), np.tan(fov / 2), num=out_h, dtype=np.float32)

    out[:, :, :2] = np.stack(np.meshgrid(x_rng, -y_rng), -1)
    Rx = np.array([[1, 0, 0], [0, np.cos(v), -np.sin(v)], [0, np.sin(v), np.cos(v)]])
    Ry = np.array([[np.cos(u), 0, np.sin(u)], [0, 1, 0], [-np.sin(u), 0, np.cos(u)]])

    R = np.dot(Ry, Rx)
    return out.dot(R.T)

def xyz_to_uv(xyz):
    x, y, z = np.split(xyz, 3, axis=-1)
    u = np.arctan2(x, z)
    c = np.sqrt(x ** 2 + z ** 2)
    v = np.arctan2(y, c)
    return np.concatenate([u, v], axis=-1)

def uv_to_XY(uv, eq_h, eq_w):
    u, v = np.split(uv, 2, axis=-1)
    X = (u / (2 * np.pi) + 0.5) * eq_w - 0.5
    Y = (-v / np.pi + 0.5) * eq_h - 0.5
    return X, Y

def eq_to_pers(eqimg, fov, u, v, out_h, out_w):
    xyz = gen_xyz(fov, u, v, out_h, out_w)
    uv  = xyz_to_uv(xyz)

    eq_h, eq_w = eqimg.shape[:2]
    XY = uv_to_XY(uv, eq_h, eq_w)
    return XY

# test the class
def case0():
    img = cv2.imread('images/360.jpg')[:,:,::-1]
    nfov = NFOV()
    center_point = np.array([0.5, .5])  # camera center point (valid range [0,1])
    ti = time.time()
    nfov.toNFOV(frame, center_point)
    print("time taken is ", time.time() - ti)

def case1():
    img = cv2.imread('images/360.jpg')[:,:,::-1]
    nfov = NFOV(v2=1)
    center_point = np.array([0.5, .5])  # camera center point (valid range [0,1])
    ti =time.time()
    nfov.toNFOV2(img, center_point)
    print("time taken is ", time.time() - ti)

def case2():
    img = cv2.imread('images/360.jpg')[...,::-1]
    nfov = NFOV(v2=1)
    # camera center point (valid range [0,1])
    center_point = [np.array([0.1, .5]),np.array([0.3, .5]),np.array([0.5, .5]),np.array([0.7, .5]),np.array([0.9, .5])]
    nfov.prepare(img)
    for c in center_point:
        ti = time.time()
        o = nfov.toNFOV3(c)
        print(time.time() - ti)

def calc_center_point(c, x, y):
    c[0] += x
    c[1] += y
    if c[0] < 0:
        c[0] = 1.0 + c[0]
    #if c[1] < 0:
    #    c[1] = 1.0 + c[1]
    if c[0] > 1:
        c[0] -= 1
    #if c[1] > 1:
    #    c[1] -= 1
    print("325", c)
    c[1] = np.clip(c[1], -0.2, 1.2)
    return c, 1

def calc_fov(fov, v):
    fov = [i+v for i in fov]
    fov = np.clip(fov, 0.05, 0.8)
    return fov

def calc_center_pointv2(c, x, y):
    c[0] += x
    c[1] += y
    print("325", c)
    #c[1] = np.clip(c[1], -0.2, 1.2)
    return c, 1

def calc_fovv2(fov, v):
    fov = [i+v for i in fov]
    #fov = np.clip(fov, 0.05, 0.8)
    return fov

def case3():
    img = cv2.imread('images/360.jpg')
    fov = [0.3,0.3]
    nfov = NFOV(600,600,v2=1,fov=fov,eqWidth=pi)
    # camera center point (valid range [0,1])
    import time
    nfov.prepare(img)
    center_point = np.array([0.5, 0.5])
    ti = time.time()
    frame = nfov.toNFOV3(center_point)
    print(time.time() - ti)
    cv2.namedWindow("T", cv2.WINDOW_NORMAL)
    s = 0.05
    while 1:
        cv2.imshow("T", frame)
        a = cv2.waitKey(30)
        ch = 0
        if a == ord('q'):
            break
        elif a in [ord('j'), ord('4')]:
            center_point, ch = calc_center_point(center_point, -s, 0)
        elif a in [ord('i'), ord('8')]:
            center_point, ch = calc_center_point(center_point, 0, -s)
        elif a in [ord('l'), ord('6')]:
            center_point, ch = calc_center_point(center_point, s, 0)
        elif a in [ord('k'), ord('2')]:
            center_point, ch = calc_center_point(center_point, 0, s)
        elif a in [ord('u'), ord('7')]:
            center_point, ch = calc_center_point(center_point, -s, -s)
        elif a in [ord('o'), ord('9')]:
            center_point, ch = calc_center_point(center_point, s, -s)
        elif a in [ord('m'), ord('1')]:
            center_point, ch = calc_center_point(center_point, -s, s)
        elif a in [ord('.'), ord('3')]:
            center_point, ch = calc_center_point(center_point, s, s)
        elif a in [ord('-'), ord('+')]:
            if a == ord('-'):
                fov = calc_fov(fov, -s)
            else:
                fov = calc_fov(fov, s)
            nfov = NFOV(600,600,v2=1,fov=fov,eqWidth=pi)
            nfov.prepare(img)
            ch = 1
        if ch:
            ti = time.time()
            frame = nfov.toNFOV3(center_point)
            print(time.time() - ti)

def case4():
    """
    https://github.com/pepepor123/equirectangular-to-cubemap/blob/master/eq2cm.py
    """
    img = cv2.imread('images/360.jpg')[:,:,::-1]
    fov = [0.7,0.7]
    nfov = GnomonicProjection(fov, dst_w=400, dst_h=400)
    nfov.prepare(img)
    # camera center point (valid range [0,1])
    import time
    center_point = [np.array([0.1, .5]),np.array([0.3, .5]),np.array([0.5, .5]),np.array([0.7, .5]),np.array([0.9, .5])]
    for c in center_point:
        ti = time.time()
        o = nfov.projectv2(c)
        print(time.time() - ti)
        Image.fromarray(o).show()

def case5():
    img = cv2.imread('images/360.jpg')
    fov = [0.7,0.7]
    nfov = GnomonicProjection(fov, dst_w=600, dst_h=600)
    nfov.prepare(img)
    center_point = np.array([0.5, 0.5])
    frame = nfov.project(center_point)
    ti = time.time()
    print(time.time() - ti)
    cv2.namedWindow("T", cv2.WINDOW_NORMAL)
    s = 0.2
    while 1:
        cv2.imshow("T", frame)
        a = cv2.waitKey(30)
        ch = 0
        if a == ord('q'):
            break
        elif a in [ord('j'), ord('4')]:
            center_point, ch = calc_center_pointv2(center_point, -s, 0)
        elif a in [ord('i'), ord('8')]:
            center_point, ch = calc_center_pointv2(center_point, 0, -s)
        elif a in [ord('l'), ord('6')]:
            center_point, ch = calc_center_pointv2(center_point, s, 0)
        elif a in [ord('k'), ord('2')]:
            center_point, ch = calc_center_pointv2(center_point, 0, s)
        elif a in [ord('u'), ord('7')]:
            center_point, ch = calc_center_pointv2(center_point, -s, -s)
        elif a in [ord('o'), ord('9')]:
            center_point, ch = calc_center_pointv2(center_point, s, -s)
        elif a in [ord('m'), ord('1')]:
            center_point, ch = calc_center_pointv2(center_point, -s, s)
        elif a in [ord('.'), ord('3')]:
            center_point, ch = calc_center_pointv2(center_point, s, s)
        elif a in [ord('-'), ord('+')]:
            if a == ord('-'):
                fov = calc_fovv2(fov, -s)
            else:
                fov = calc_fovv2(fov, s)
            nfov = GnomonicProjection(fov, dst_w=600, dst_h=600)
            nfov.prepare(img)
            ch = 1
        if ch:
            ti = time.time()
            frame = nfov.project(center_point)
            print(time.time() - ti)
if __name__ == '__main__':
    case5()
