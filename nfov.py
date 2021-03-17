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

class NFOV():
    def __init__(self, height=400, width=400, v2=0):
        self.FOV = [0.6, 0.6]
        self.PI = pi
        self.PI_2 = pi * 0.5
        self.PI2 = pi * 2.0
        self.height = height
        self.width = width
        self.v2 = v2
        if self.v2:
            self.screen_points = self._get_screen_img_v2()
        else:
            self.screen_points = self._get_screen_img()

    def _get_coord_rad(self, isCenterPt, center_point=None):
        return (center_point * 2 - 1) * np.array([self.PI, self.PI_2]) \
            if isCenterPt \
            else \
            (self.screen_points * 2 - 1) * np.array([self.PI, self.PI_2]) * (
                np.ones(self.screen_points.shape) * self.FOV)

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
        lon = (lon / self.PI + 1.) * 0.5

        return np.dstack([lon, lat])

    def _calcSphericaltoGnomonic_v3(self):
        x = self.x
        y = self.y

        rou = self.rou
        c = self.c
        sin_c = self.sin_c
        cos_c = self.cos_c
        cp = self.cp/2
        lat = np.arcsin(cos_c * np.sin(cp[1]) + (y * sin_c * np.cos(cp[1])) / rou)
        lon = cp[0] + np.arctan2(x * sin_c, rou * np.cos(cp[1]) * cos_c - y * np.sin(cp[1]) * sin_c)

        lat = (lat / self.PI_2 + 1.) * 0.5
        lon = (lon / self.PI + 1.) * 0.5

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

        mask = (uf < 0) | (uf >= (self.frame_width - 2)) | (vf < 0) | (vf > (self.frame_height - 2))
        uf[mask] = 0
        vf[mask] = 0

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
# test the class
def case0():
    import imageio as im
    img = im.imread('images/360.jpg')
    nfov = NFOV()
    center_point = np.array([0.5, .5])  # camera center point (valid range [0,1])
    import time
    ti = time.time()
    nfov.toNFOV(frame, center_point)
    print("time taken is ", time.time() - ti)

def case1():
    import imageio as im
    import time
    img = im.imread('images/360.jpg')
    nfov = NFOV(v2=1)
    center_point = np.array([0.5, .5])  # camera center point (valid range [0,1])
    ti =time.time()
    nfov.toNFOV2(frame, center_point)
    print("time taken is ", time.time() - ti)

def case2():
    import imageio as im
    img = im.imread('images/360.jpg')
    nfov = NFOV(v2=1)
    # camera center point (valid range [0,1])
    import time
    center_point = [np.array([0.1, .5]),np.array([0.3, .5]),np.array([0.5, .5]),np.array([0.7, .5]),np.array([0.9, .5])]
    nfov.prepare(img)
    for c in center_point:
        ti = time.time()
        o = nfov.toNFOV3(c)
        print(time.time() - ti)

def calc_center_point(c, x, y):
    c[0] += x
    c[1] += y
    c = np.clip(c, 0.1, 0.9)
    return c, 1

def case3():
    import imageio as im
    img = im.imread('images/360.png')[:,:,::-1]
    nfov = NFOV(v2=1)
    # camera center point (valid range [0,1])
    import time
    nfov.prepare(img)
    center_point = np.array([0.5, 0.5])
    ti = time.time()
    frame = nfov.toNFOV3(center_point)
    print(time.time() - ti)
    cv2.namedWindow("T", cv2.WINDOW_NORMAL)
    s = 0.02
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
        if ch:
            ti = time.time()
            frame = nfov.toNFOV3(center_point)
            print(time.time() - ti)

if __name__ == '__main__':
    case3()
