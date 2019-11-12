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

def calcHomography(u, v):
    """DLT algo"""

    # get assemable matrix
    mat = calc_corresp(u, v)
    
    # svd composition
    u, s, v = np.linalg.svd(mat)
    
    # reshape
    h = v[-1].reshape(3, 3)
    
    # norm
    h = h / h.item(8)
    return h

def calcHomographyLinear(u, v):
    """Linear"""

    # get assemable matrix
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
    img[0,0,0] = 0
    img[0,0,1] = 0
    img[0,0,2] = 0
    mask = (z_t[:, 0] > w-1) | (z_t[:, 0] < 0) | (z_t[:, 1] > h-1) | (z_t[:, 1] < 0)
    z_t[mask, 0:2]  = 0
    tmp = img[z_t[:,1],z_t[:,0],:]
    img_n = tmp.reshape(mh, mw, 3)
    return img_n

def bilinear(z_t, img, h, w, mh, mw):
    z_t = z_t.T
    img[0,0,0] = 0
    img[0,0,1] = 0
    img[0,0,2] = 0
    mask = (z_t[:, 0] > w-1) | (z_t[:, 0] < 0) | (z_t[:, 1] > h-1) | (z_t[:, 1] < 0)
    z_t[mask, 0:2]  = 0
    z_ti = z_t.astype(np.int32)
    z_t_f = z_t - z_ti

    img0 = img[z_ti[:,1],z_ti[:,0],:] * (1 - z_t_f[:,0:1]) + img[z_ti[:,1],z_ti[:,0]+1,:] * (z_t_f[:,0:1])
    img1 = img[z_ti[:,1]+1,z_ti[:,0],:] * (1 - z_t_f[:,0:1]) + img[z_ti[:,1]+1,z_ti[:,0]+1,:] * (z_t_f[:,0:1])
    img_n = img0 * ( 1 - z_t_f[:,1:2]) + img1 * (z_t_f[:,1:2])
    return img_n.reshape(mh, mw, 3)

convertfunc = {'nn':nearestNeighbor, 'bilinear':bilinear}

def wrapPerspective(img, H, convert='nn'):
    h, w, c = img.shape
    bnd = np.array([[0, w-1, w-1, 0],
                   [0, 0,   h-1, h-1],
                   [1., 1,   1,   1]])
    bnd_n = H @ bnd
    bnd_n /= bnd_n[-1,:]
    # find max boundary
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
    img_n = convertfunc[convert](z_t, img, h, w, max_h, max_w)
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