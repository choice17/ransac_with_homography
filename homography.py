import cv2
import numpy as np

def calc_corresp(u, v):
    # 2N * 9
    a = np.array([[-u[0,0], -u[0,1], -1, 0, 0, 0, u[0,0]*v[0,0], u[0,1]*v[0,0], v[0,0]],
                  [0, 0, 0, -u[0,0], -u[0,1], -1, u[0,0]*v[0,1], u[0,1]*v[0,1], v[0,1]],
                  [-u[1,0], -u[1,1], -1, 0, 0, 0, u[1,0]*v[1,0], u[1,1]*v[1,0], v[1,0]],
                  [0, 0, 0, -u[1,0], -u[1,1], -1, u[1,0]*v[1,1], u[1,1]*v[1,1], v[1,1]],
                  [-u[2,0], -u[2,1], -1, 0, 0, 0, u[2,0]*v[2,0], u[2,1]*v[2,0], v[2,0]],
                  [0, 0, 0, -u[2,0], -u[2,1], -1, u[2,0]*v[2,1], u[2,1]*v[2,1], v[2,1]],
                  [-u[3,0], -u[3,1], -1, 0, 0, 0, u[3,0]*v[3,0], u[3,1]*v[3,0], v[3,0]],
                  [0, 0, 0, -u[3,0], -u[3,1], -1, u[3,0]*v[3,1], u[3,1]*v[3,1], v[3,1]]]).astype(np.float32)
    return a

def calc_correspLinear(u, v):
    # 2N * 8 see https://scm_mos.gitlab.io/vision/homography-matrix/
    a = np.array([[u[0,0], u[0,1], 1, 0, 0, 0, -u[0,0]*v[0,0], -u[0,1]*v[0,0]],
                  [0, 0, 0, u[0,0], u[0,1], 1, -u[0,0]*v[0,1], -u[0,1]*v[0,1]],
                  [u[1,0], u[1,1], 1, 0, 0, 0, -u[1,0]*v[1,0], -u[1,1]*v[1,0]],
                  [0, 0, 0, u[1,0], u[1,1], 1, -u[1,0]*v[1,1], -u[1,1]*v[1,1]],
                  [u[2,0], u[2,1], 1, 0, 0, 0, -u[2,0]*v[2,0], -u[2,1]*v[2,0]],
                  [0, 0, 0, u[2,0],u[2,1], 1, -u[2,0]*v[2,1], -u[2,1]*v[2,1]],
                  [u[3,0], u[3,1], 1, 0, 0, 0, -u[3,0]*v[3,0], -u[3,1]*v[3,0]],
                  [0, 0, 0, u[3,0], u[3,1], 1, -u[3,0]*v[3,1], -u[3,1]*v[3,1]]]).astype(np.float32)

    b = np.array([[v[0,0], v[0,1], v[1,0], v[1,1], v[2,0], v[2,1], v[3,0], v[3,1]]]).astype(np.float32).T
    return a, b

def calc_correspCollective(u, v):
    mx, nx = u.shape
    a = np.zeros((mx, 2 * 9), dtype=np.float32)
    a[:, 0] = -u[:, 0]
    a[:, 1] = -u[:, 1]
    a[:, 2] = -1
    a[:, 6] = u[:, 0] * v[:, 0]
    a[:, 7] = u[:, 1] * v[:, 0]
    a[:, 8] = v[:, 0]
    a[:, 12] = -u[:, 0]
    a[:, 13] = -u[:, 1]
    a[:, 14] = -1
    a[:, 15] = u[:, 0] * v[:, 1]
    a[:, 16] = u[:, 1] * v[:, 1]
    a[:, 17] = v[:, 1]
    a = a.reshape(mx * 2, 9)
    return a

def calc_correspLinearCollective(u, v):
    # 2N * 8 see https://scm_mos.gitlab.io/vision/homography-matrix/
    mx, nx = u.shape
    a = np.zeros((mx, 2 * 8), dtype=np.float32)
    b = np.zeros((mx, 2), dtype=np.float32)
    a[:, 0] = u[:, 0]
    a[:, 1] = u[:, 1]
    a[:, 2] = 1
    a[:, 6] = -u[:, 0] * v[:, 0]
    a[:, 7] = -u[:, 1] * v[:, 0]
    a[:, 11] = u[:, 0]
    a[:, 12] = u[:, 1]
    a[:, 13] = 1
    a[:, 14] = -u[:, 0] * v[:, 1]
    a[:, 15] = -u[:, 1] * v[:, 1]

    b[:, 0] = v[:, 0]
    b[:, 1] = v[:, 1]

    a = a.reshape(mx * 2, 8)
    b = b.reshape(mx * 2, 1)
    return a, b

def calcHomography(u, v, collective=False):
    """DLT algo"""

    # get assemable matrix
    if collective:
        mat = calc_correspCollective(u, v)
    else:
        mat = calc_corresp(u, v)
    
    # svd composition
    u, s, v = np.linalg.svd(mat)
    
    # reshape
    h = v[-1].reshape(3, 3)
    
    # norm
    h = h / h.item(8)
    return h

def calcHomographyLinear(u, v, collective=False):
    """Linear"""

    # get assemable matrix
    if collective:
        A, b = calc_correspLinearCollective(u, v)
    else:
        A, b = calc_correspLinear(u, v)
    
    # solver linear AH = b, H = (A.T * A)^-1 * A.T * b
    h = np.linalg.inv(A.T @ A) @ (A.T @ b)
    
    H = np.array([[h.item(0), h.item(1), h.item(2)],
                  [h.item(3), h.item(4), h.item(5)],
                  [h.item(6), h.item(7), 1]])
    return H


def nearestNeighbor(z_t, img, h, w, mh, mw):
    # rounding
    z_t = (z_t + 0.5).astype(np.int32).T
    chn = img.shape[2]
    img[0,0,0] = 0
    img[0,0,1] = 0
    img[0,0,2] = 0
    if chn == 4:
        img[0,0,3] = 0
    mask = (z_t[:, 0] > w-1) | (z_t[:, 0] < 0) | (z_t[:, 1] > h-1) | (z_t[:, 1] < 0)
    z_t[mask, 0:2]  = 0
    tmp = img[z_t[:,1],z_t[:,0],:]
    img_n = tmp.reshape(mh, mw, chn)
    return img_n

def bilinear(z_t, img, h, w, mh, mw):
    z_t = z_t.T
    chn = img.shape[2]
    img[0,0,0] = 0
    img[0,0,1] = 0
    img[0,0,2] = 0
    if chn == 4:
        img[0,0,3] = 0
    mask = (z_t[:, 0] > w-1) | (z_t[:, 0] < 0) | (z_t[:, 1] > h-1) | (z_t[:, 1] < 0)
    z_t[mask, 0:2]  = 0
    z_ti = z_t.astype(np.int32)
    z_t_f = z_t - z_ti
    img0 = img[z_ti[:,1],z_ti[:,0],:] * (1 - z_t_f[:,0:1]) + img[z_ti[:,1],z_ti[:,0]+1,:] * (z_t_f[:,0:1])
    img1 = img[z_ti[:,1]+1,z_ti[:,0],:] * (1 - z_t_f[:,0:1]) + img[z_ti[:,1]+1,z_ti[:,0]+1,:] * (z_t_f[:,0:1])
    img_n = img0 * ( 1 - z_t_f[:,1:2]) + img1 * (z_t_f[:,1:2])
    return img_n.reshape(mh, mw, chn)

convertfunc = {'nn':nearestNeighbor, 'bilinear':bilinear}

def wrapPerspective(img, H, convert='nn', boundary=0, crop=True):
    h, w, c = img.shape
    bnd = np.array([[0, w-1, w-1, 0],
                   [0, 0,   h-1, h-1],
                   [1., 1,   1,   1]])
    bnd_n = H @ bnd
    bnd_n /= bnd_n[-1,:]
    # find max boundary
    if boundary:
        max_x = int(np.max(bnd_n[0,:]))
        min_x = max(int(np.min(bnd_n[0,:])),0)
        max_y = int(np.max(bnd_n[1,:]))
        min_y = max(int(np.min(bnd_n[1,:])),0)
        max_w = max_x - min_x + 1
        max_h = max_y - min_y + 1
    else:
        max_x = int(np.max(bnd_n[0,:]))
        min_x = int(np.min(bnd_n[0,:]))
        max_y = int(np.max(bnd_n[1,:]))
        min_y = int(np.min(bnd_n[1,:]))
        max_w = max_x - min_x + 1
        max_h = max_y - min_y + 1

    # Construct meshgrid for transformation
    x = np.linspace(min_x, max_x, max_w)
    y = np.linspace(min_y, max_y, max_h)
    xv, yv = np.meshgrid(x, y)
    z = np.dstack([xv, yv, np.ones((max_h, max_w))]).reshape([max_w * max_h , 3]).T

    # backward wrapping
    invH = np.linalg.inv(H) 
    z_t = invH @ z
    z_t /= z_t[-1,:]

    # interpolation
    img_n = None
    if crop:
        img_n = convertfunc[convert](z_t, img, h, w, max_h, max_w)
    else:
        img_n = np.zeros((max_h, max_w, 3),dtype=np.float32)
        img_t = convertfunc[convert](z_t, img, h, w, max_h, max_w)
        img_n[min_y:min_y+max_h,min_x:min_x+max_w,:] = img_t
    return img_n, min_x, min_y

def wrapPerspectiveScan(img, H, res, convert='nn'):
    imgh, imgw, _ = img.shape
    h, w = res
    min_x = 0
    min_y = 0
    max_x = w
    max_y = h
    max_w = w
    max_h = h

    # Construct meshgrid for transformation
    x = np.linspace(min_x, max_x, max_w)
    y = np.linspace(min_y, max_y, max_h)
    xv, yv = np.meshgrid(x, y)
    z = np.dstack([xv, yv, np.ones((max_h, max_w))]).reshape([max_w * max_h , 3]).T

    # backward wrapping
    invH = np.linalg.inv(H)
    z_t = invH @ z
    z_t /= z_t[-1,:]

    # interpolation
    img_n = convertfunc[convert](z_t, img, h, w, max_h, max_w)
    return img_n, min_x, min_y

def transformImage(img, u, v, box=None, method='bilinear'):
    """ transformImage
    @brief: Homography perspective transform from u to v
    @[in] img: numpy image MxNx3
    @[in] u:   src 4 corners 3xn
    @[in] v:   dest 4 corners 3xn
    @[in] method: interpolation method:nn/bilinear
    @[out] o:  transformed image
    """
    H = calcHomographyLinear(u.T[:,:2], v.T[:,:2])
    if box is None:
        imgn, mx, my = wrapPerspective(img, H, convert=method)
    else:
        imgn, mx, my = wrapPerspectiveScan(img, H, box, convert=method)
    imgn = imgn.astype(np.uint8)
    sx = int(v[0,0]-mx); sy = int(v[1,0]-my)
    ex = int(v[0,2]-mx); ey = int(v[1,2]-my)
    return imgn[sy:ey+1, sx:ex+1,:]

def transformImageH(img, H, method='bilinear'):
    """ transformImage
    @brief: Homography perspective transform by H
    @[in] img: numpy image MxNx3
    @[in] H: transform matrix
    @[in] method: interpolation method:nn/bilinear
    @[out] o:  transformed image
    """

    imgn, mx, my = wrapPerspective(img, H, convert=method, boundary=0, crop=True)
    if imgn.shape[2] == 3:
        return imgn.astype(np.uint8), mx, my
    return imgn, mx, my

class BLENDDIR(object):
    LEFT = 0
    RIGHT = 1
    TOP = 2
    DOWN = 3
 
def addAlpha(img, method="Rate", rate=0.2, direction=BLENDDIR.LEFT, alphaOnly=False):
    h, w, c = img.shape
    rate += 1e-10
    if not alphaOnly:
        imgn = np.zeros((h, w, c+1),dtype=np.float32)
        imgn[:, :, :c] = img
        if method=='Rate':
            print(rate)
            imgn[:, :, c] = rate
        elif method=='Gradient':
            if direction==BLENDDIR.LEFT:
                x = np.linspace(0, w-1, w)
                y = np.linspace(0, h-1, h)
                xx, yy = np.meshgrid(x,y)
                br = (xx + yy)/(w + h) * 0.5
            imgn[:, :, c] = br
            print('not implement yet')
        return imgn
    alpha = np.zeros((h, w, 1),dtype=np.float32)
    if method=='Rate':
        print(rate)
        alpha[:, :, 0] = rate
    elif method=='Gradient':
        rate = 0.5
        if direction==BLENDDIR.LEFT:
            x = np.linspace(0, w-1, w)
            y = np.linspace(0, h-1, h)
            xx, yy = np.meshgrid(x,y)
            br = (xx + yy)/(w + h) * 0.5
        elif direction==BLENDDIR.RIGHT:
            x = np.linspace(w-1, 0, w)
            y = np.linspace(h-1, 0, h)
            xx, yy = np.meshgrid(x,y)
            br = (xx + yy)/(w + h) * 0.5
        alpha[:, :, 0] = br
        print('not implement yet')
    return alpha

def stitchPanorama(imgQ, imgT, H, method='bilinear', blending=False, blendrate=0.2):
    """ stitchPanorama
    @brief: stitch two images by transformation matrix
    @[in] imgQ: numpy query image MxNx3
    @[in] imgT: numpy transform image mxnx3
    @[in] H: transform matrix [3x3]
    @[in] method: interpolation method:nn/bilinear
    @[out] o:  transformed image
    """
    if blending:
        imgT = addAlpha(imgT, method=blending, rate=blendrate)
    img_t, mx, my = transformImageH(imgT, H)
    ht, wt, ct = img_t.shape
    hq, wq, cq = imgQ.shape

    tsx = 0; tsy = 0; tex = wt-1; tey = ht-1
    qsx = 0; qsy = 0; qex = wq-1; qey = hq-1
    fsx = min(0, mx); fsy = min(0, my)
    fex = max(wq-1, mx+wt); fey = max(hq-1, my+ht)
    fw = fex - fsx + 1; fh = fey - fsy + 1
    
    if (mx < 0) and (my < 0):
        tex = wt-1; tey = ht-1
        qsx = -mx; qsy = -my; qex = -mx+wq-1; qey = -my+hq-1
    elif (mx < 0):
        tsy = my; tey = my+ht-1
        qsx = -mx; qex = -mx+wq-1
    elif (mx >= 0) and (my < 0):
        tsx = mx; tex = mx+wt-1
        qsy = -my; qey = -my+hq-1
    elif (mx >= 0):
        tsx = mx; tsy = my; tex = mx+wt-1; tey = my+ht-1
    fw = max(tex+1, qex+1)
    fh = max(tey+1, qey+1)
    if blending:
        imgn = np.zeros((fh,fw,ct), dtype=np.float32)
        # do blending here
        imgn[qsy:qey+1,qsx:qex+1,:3] = imgQ[:,:,:3].astype(np.float32)
        imgn[:,:, 3] += 1e-10
        if blending == 'Rate':
            imgn[qsy:qey+1,qsx:qex+1, 3] = 1 + 1e-10 - blendrate
        else:
            imgn[qsy:qey+1,qsx:qex+1, 3] = 1
        base = (imgn[tsy:tey+1,tsx:tex+1,3:4] + img_t[:,:,3:4])
        imgn[tsy:tey+1,tsx:tex+1,:3] = \
        ((imgn[tsy:tey+1,tsx:tex+1,3:4]/base) * imgn[tsy:tey+1,tsx:tex+1,:3] + (img_t[:,:,3:4]/base) * img_t[:,:,:3])
        return imgn[:,:,:3].astype(np.uint8)
    imgn = np.zeros((fh,fw,ct), dtype=np.uint8)
    imgn[tsy:tey+1,tsx:tex+1,:] = img_t
    imgn[qsy:qey+1,qsx:qex+1,:] = imgQ
    return imgn

def example0():
    imgFile = 'notebook.jpg'
    img = cv2.imread(imgFile)
    u = np.array([[50,470,600.,90],
                  [50,40,300.,360],
                  [1, 1,  1, 1.]])
    v = np.array([[0, 400, 400., 0],
                  [0, 0,   780., 780],
                  [1,  1,  1,  1]])
    H = calcHomographyLinear(u.T[:,:2], v.T[:,:2])

    imgn, mx, my = wrapPerspective(img, H, convert='bilinear')
    #imgn = cv2.warpPerspective(img, H, (2000, 1000))
    imgn = imgn.astype(np.uint8)
    h, w, _ = img.shape
    nh, nw, _ = imgn.shape
    WIN = "T"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    imgall = np.zeros((nh, w+nw,3),dtype=np.uint8)
    imgall[:h,:w,:] = img
    imgall[:,w:w+nw,:] = imgn
    while (1):
        cv2.imshow("T", imgall)
        k = cv2.waitKey(0)
        if (k == ord('q')):
            exit(0)

def example1():
    pass

def main():
    example0()
    example1()

if __name__ == '__main__':
    main()