import numpy as np
import cv2
from PIL import Image

def info(*args):
    print("[INFO] ", *args)

def trc(*args):
    #print("[INFO] ", *args)
    pass

def debug(*args):
    pass

def sperical_transform(dst_resoln, src, src_rx, src_ry, cx, cy):

    dh, dw = dst_resoln
    sh, sw, _ = src.shape
    dst_img = np.zeros((dh, dw, 3), dtype=np.uint8)
    dst_coord_x, dst_coord_y = np.meshgrid(
        (np.arange(dw)/dw - 0.5)* np.pi,
        (np.arange(dh)/dh - 0.5)* np.pi)

    src_coord_x = np.tan(dst_coord_x) * src_rx
    src_coord_y = np.tan(dst_coord_y) * np.cos(dst_coord_x) * src_ry
    info("param", src_rx, src_ry, cx, cy)
    debug(0, dst_coord_x[0,::50])
    debug(0, dst_coord_y[0,::50])
    src_coord_x += cx
    src_coord_y += cy
    debug(1, src_coord_x[0,::10])
    np.clip(src_coord_x, 0, sw-1, out=src_coord_x)
    np.clip(src_coord_y, 0, sh-1, out=src_coord_y)
    debug(2, src_coord_x[0,::10])
    debug("src x", src_coord_x.shape , "min", src_coord_x.min(), "max", src_coord_x.max())
    debug("src y", src_coord_y.shape, "min", src_coord_y.min(), "max", src_coord_y.max())
    src_coord_x = src_coord_x.astype(np.int32)
    src_coord_y = src_coord_y.astype(np.int32)
    dst_img = src[src_coord_y, src_coord_x, :]
    return dst_img

def sperical_transform2(dst_resoln, src, src_rx, src_ry, cx, cy, dst_angle_x=np.pi,dst_angle_y=np.pi):

    dh, dw = dst_resoln
    sh, sw, _ = src.shape
    dst_img = np.zeros((dh, dw, 3), dtype=np.uint8)
    src[0,0,0] = src[0,0,1]= src[0,0,2] = 0 

    #src_rx = src_ry

    #hor_straight = src_rx / scale_rx
    #src_zoom = scale_rx / scale_ry
    #src_x_scaler = src_rx / (magic_r * hor_straight)
    #src_y_scaler = src_rx / ( )

    # step 1. define step size -pi/2 to pi/2
    dst_coord_x, dst_coord_y = np.meshgrid(
        (np.arange(dw)/dw - 0.5)* dst_angle_x,
        (np.arange(dh)/dh - 0.5)* dst_angle_y)

    # step 2. calcu spherical tranform coord 
    # dstangle map => 3d sphere xyz => src angle map
    # src_x = 2 / pi * (arccos_cosAcosB) * cos(arctan2(sinB/sinAcosB))
    # src_y = 2 / pi * (arccos_cosAcosB) * sin(arctan2(sinB/sinAcosB))

    cosB = np.cos(dst_coord_y)
    # range = [-pi,pi]
    arctan_sinB_sinAcosB = np.arctan2(
        np.sin(dst_coord_y),
        (np.sin(dst_coord_x)*np.cos(dst_coord_y) + 1e-10))
    # range = [0,pi/2]
    arccos_cosAcosB = np.arccos(cosB * np.cos(dst_coord_x))

    debug(((((arctan_sinB_sinAcosB))[::50,::100] * 100)).astype(np.int32)/100.0)
    
    # Method I.
    src_coord_x = 2. / np.pi * arccos_cosAcosB * np.cos(arctan_sinB_sinAcosB)
    src_coord_y = 2. / np.pi * arccos_cosAcosB * np.sin(arctan_sinB_sinAcosB)
    # Method II.
    #src_coord_x = np.sin(arccos_cosAcosB) * np.cos(arctan_sinB_sinAcosB)
    #src_coord_y = np.sin(arccos_cosAcosB) * np.sin(arctan_sinB_sinAcosB)


    # step 3. multi by src spherical param
    src_coord_x *= src_rx
    src_coord_y *= src_ry

    info("param", src_rx, src_ry, cx, cy)

    # step 4. offset translation
    src_coord_x += cx
    src_coord_y += cy


    # step 5. clipping
    mask = (src_coord_x < 0) | (src_coord_x >= (sw - 2)) | (src_coord_y < 0) | (src_coord_y > (sh - 2))
    src_coord_x[mask] = 0
    src_coord_y[mask] = 0
    #np.clip(src_coord_y, 0, sh-2, out=src_coord_y)
    src_coord_xf = src_coord_x
    src_coord_yf = src_coord_y
    src_coord_x = src_coord_x.astype(np.int32)
    src_coord_y = src_coord_y.astype(np.int32)

    # step 6. bilinear interpolation wrapping
    for i in range(3):
        x_f = src_coord_xf - src_coord_x
        y_f = src_coord_yf - src_coord_y
        img0 = src[src_coord_y,src_coord_x,i] * (1 - x_f) + src[src_coord_y,src_coord_x+1,i] * x_f
        img1 = src[src_coord_y+1,src_coord_x,i] * (1 - x_f) + src[src_coord_y+1,src_coord_x+1,i] * x_f
        dst_img[:,:,i] = img0 * ( 1 - y_f) + img1 * (y_f)

    # step 6. nearest neighbour wrapping
    # src_coord_x = (src_coord_x + 0.5).astype(np.int32)
    # src_coord_y = (src_coord_y + 0.5).astype(np.int32)
    # dst_img = src[src_coord_y,src_coord_x,:]

    return dst_img

def sperical_transform2(dst_resoln, src, src_rx, src_ry, cx, cy, magic_r=0, scale_rx=0, scale_ry=0, dst_angle_x=np.pi,dst_angle_y=np.pi):
    # TODO implement magic r scale_rx scale_ry
    #src_rx = src_ry

    #hor_straight = src_rx / scale_rx
    #src_zoom = scale_rx / scale_ry
    #src_x_scaler = src_rx / (magic_r * hor_straight)
    #src_y_scaler = src_rx / ( )

    # step 1. define step size -pi/2 to pi/2

    # step 2. calcu spherical tranform coord 
    # dstangle map => 3d sphere xyz => src angle map
    # src_x = 2 / pi * (arccos_cosAcosB) * cos(arctan2(sinB/sinAcosB))
    # src_y = 2 / pi * (arccos_cosAcosB) * sin(arctan2(sinB/sinAcosB))

    # step 3. multi by src spherical param

    # step 4. offset translation

    # step 5. clipping/ boudnary solution

    # step 6. coord wrapping (nearest neighbour/ bilinear)

    pass

def example1():
    fn = "images/sphere_image.jpg"
    src = cv2.imread(fn)[:,:,::-1]
    info("src image ", src.shape)
    dst_resoln = [1200,2400]
    src_rx = 240
    src_ry = 240
    cx = src.shape[1] / 2
    cy = src.shape[0] / 2
    dst_img = sperical_transform2(dst_resoln, src, src_rx, src_ry, cx, cy, dst_angle_x=np.pi,dst_angle_y=np.pi*0.5)
    cv2.imwrite("tansformed.jpg", dst_img)
    Image.fromarray(dst_img).show()

def example2():
    fn = "images/indoor_fisheye.jpg"
    # 4000x2666
    src = cv2.imread(fn)[:,:,::-1]
    info("src image ", src.shape)
    dst_resoln = [1200,3600]
    src_rx = 1250
    src_ry = 1250
    cx = src.shape[1] / 2
    cy = src.shape[0] / 2
    dst_img = sperical_transform2(dst_resoln, src, src_rx, src_ry, cx, cy, dst_angle_x=np.pi,dst_angle_y=np.pi)
    cv2.imwrite("tansformed.jpg", dst_img)
    Image.fromarray(dst_img).show()

def example3():
    fn = "images/indoor_crop_fisheye.jpg"
    # 2141x1457
    src = cv2.imread(fn)[:,:,::-1]
    info("src image ", src.shape)
    dst_resoln = [1200,2000]
    src_rx = 1250
    src_ry = 1250
    cx = src.shape[1] / 2
    cy = src.shape[0] / 2
    dst_img = sperical_transform2(dst_resoln, src, src_rx, src_ry, cx, cy, dst_angle_x=np.pi,dst_angle_y=np.pi)
    cv2.imwrite("tansformed.jpg", dst_img)
    Image.fromarray(dst_img).show()

def example4():
    fn = "indoor_sq_fisheye.jpg"
    # 1789x1781
    src = cv2.imread(fn)[:,:,::-1]
    info("src image ", src.shape)
    dst_resoln = [1200 ,3600]
    src_rx = 1250
    src_ry = 1250
    cx = src.shape[1] / 2
    cy = src.shape[0] / 2
    dst_img = sperical_transform2(dst_resoln, src, src_rx, src_ry, cx, cy, dst_angle_x=np.pi,dst_angle_y=np.pi*0.75)
    cv2.imwrite("tansformed.jpg", dst_img)
    Image.fromarray(dst_img).show()

def main():

    # spherical transform from fisheye image
    # example1()

    # spherical transform from croped image
    example2()

    #example3()

    # cylinder transform

if __name__ == "__main__":
    main()