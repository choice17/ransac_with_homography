import cv2
import numpy as np
import sys
import tkinter as tk
import tkinter.filedialog as F
import threading as pthread
import os
import time
from homography import transformImage
from ransac import stitching
import signal

debugLvl = 0.5 

WIN = 'app'
WINT = 'transform'
TIME_MAX = 5
H, W, C = 400, 500, 3

class SIZE_S(object):
    A4 = 297 * 4, 210 * 4
    F4 = 330 * 4, 210 * 4
    CARD = 53.98 * 4, 85.6 * 4 


def INFO(*args):
    if debugLvl >= 0:
        print("[INFO] ", *args)

def INFOH(*args):
    if debugLvl >= 0.5:
        print("[INFOH] ", *args)

def WARN(*args):
    if debugLvl >= 0:
        print("[WARN] ", *args)

def DEBUG(*args):
    if debugLvl >= 2:
        print("[DEBUG] ", *args)

def ERR(*args):
    if debugLvl >= 0:
        print("[ERR] ", *args)

def calcFixedAspectRatio(win_dim, tar_dim):
    """
    @param: win_dim current window dim (width, height)
    @param: tar_dim target image dim (width, height)
    """
    w, h = win_dim
    tw, th = tar_dim
    if w > h:
        h_r = h ; w_r = int(h * (tw/ th))
        if w_r > w:
            h_r = int(w * (th / tw))
            w_r = w
    else: # h > w:
        w_r = w ; h_r = int(w * (th / tw))
        if h_r > h:
            h_r = h
            w_r = int(h * (tw / th))
    sz = (w_r, h_r)
    if 0 in sz:
        return 0, sz, 0, 0, 0, 0
    else:
        sx = (w >> 1) - ((w_r + 1) >> 1) ; ex = sx + w_r
        sy = (h >> 1) - ((h_r + 1) >> 1) ; ey = sy + h_r
        if sx < 0: ex -= sx ; sx = 0
        if sy < 0: ey -= sy ; sy = 0
        if ex > w: sx -= w - ex ; ex = w
        if ey > h: sy -= h - ey ; ey = h
    return 1, sz, sx, sy, ex, ey

def run_resize_image(image, cvwin, ww, hh, lock=0):
    x, y, w, h = cv2.getWindowImageRect(cvwin)
    if (w > 0) and h > 0:
        suc, sz, sx, sy, ex, ey = calcFixedAspectRatio([w, h], [ww, hh])
        if suc:
            #DEBUG(suc, sz, sx, sy, ex, ey, (ww, hh), (x, y, w, h), image.shape, cvwin)
            img = cv2.resize(image, sz, interpolation=cv2.INTER_LINEAR)
            if lock:
                image = img
                cv2.resizeWindow(cvwin, sz[0], sz[1])
            else:
                image = np.zeros((h, w, 3), dtype=np.uint8)
                image[sy:ey, sx:ex, :] = img
    return image, [w,h]

class Control(object):
    SCANNER = 0
    PANORAMA = 1
    PANO_CNT_TH = 2

    def __init__(self):
        self.app_quit = 0
        self.cv_quit = 0
        self.cvt_quit = 0
        self.img_name = None
        self.img = None
        self.oimg = None
        self.img2 = None
        self.mode = self.SCANNER
        self.pano_cnt = 0

class App(object):
    def __init__(self):
        self.quit = 0
        self.cv_app_up = 0

    def openScanner(self):
        global cb
        name = F.askopenfilename(initialdir=os.getcwd(),
                               filetypes =(("Image Files","*.jpg"),("All Files","*.*"),),
                               title = "Choose a file."
                               )
        INFOH("-%s-" % name)
        if os.path.exists(name):
        #Using try in case user types in unknown file or closes without choosing a file.
            try:
                g_ctx.img_name = name
                g_ctx.img = cv2.imread(name)
                    #print(UseFile.read())
            except:
                WARN("No file exists")
            if (self.cv_app_up):
                g_ctx.cv_quit = 1
                self.cv_app.join()
                self.cv_app_up = 0
                DEBUG("turn off image")

            cb.cur_time = 0
            g_ctx.cv_quit = 0
            self.cvapp = CvApp()
            self.cv_app = pthread.Thread(target=self.cvapp.run, args=(cb,))
            self.cv_app.start()
            self.cv_app_up = 1
            INFOH("start new session")
        else:
            ERR("Cannot reach the file %s" % name)

    def openPanorama(self):
        global cb, g_ctx
        name = F.askopenfilename(initialdir=os.getcwd(),
                               filetypes =(("Image Files","*.jpg"),("All Files","*.*"),),
                               title = "Choose a file."
                               )
        INFOH("-%s-" % name)
        if os.path.exists(name):
        #Using try in case user types in unknown file or closes without choosing a file.
            try:
                if (g_ctx.pano_cnt == 0):
                    g_ctx.img_name = name
                    g_ctx.img = cv2.imread(name)
                    INFO("Query Image Selected")
                else:
                    g_ctx.img2_name = name
                    g_ctx.img2 = cv2.imread(name)
                    INFO("Transform Image Selected")
                g_ctx.pano_cnt = (g_ctx.pano_cnt + 1) % Control.PANO_CNT_TH
                    #print(UseFile.read())
            except:
                WARN("No file exists")
            """if (self.cv_app_up):
                g_ctx.cv_quit = 1
                self.cv_app.join()
                self.cv_app_up = 0
                DEBUG("turn off image")

            cb.cur_time = 0
            g_ctx.cv_quit = 0
            self.cvapp = CvApp()
            self.cv_app = pthread.Thread(target=self.cvapp.run, args=(cb,))
            self.cv_app.start()
            self.cv_app_up = 1
            """
            INFOH("start new session")
        else:
            ERR("Cannot reach the file %s" % name)

    def open(self):
        global g_ctx
        if g_ctx.mode == Control.SCANNER:
            self.openScanner()
        elif g_ctx.mode == Control.PANORAMA:
            self.openPanorama()
        return

    def save(self):
        if g_ctx.oimg is not None:
            filename =  F.asksaveasfilename(initialdir = os.getcwd(),title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
            try:
                cv2.imwrite(filename, g_ctx.oimg)
            except:
                cv2.imwrite(filename + '.jpg', g_ctx.oimg)
            INFO("save to %s!" % filename)
        else:
            INFO("Please apply transform image")

    def exit(self):
        self.win.destroy()

    def scanner(fmt):
        global SCANNERV, g_ctx
        g_ctx.mode = Control.SCANNER
        SCANNERV = np.array([[0, int(fmt[1]), int(fmt[1]), 0],
                      [0, 0,   int(fmt[0]), int(fmt[0])],
                      [1,  1,  1,  1]])
        WARN("Scanner mode selected!")

    def panorama(self):
        global g_ctx
        g_ctx.mode = Control.PANORAMA
        g_ctx.pano_cnt = 0
        WARN("Panorama mode selected!")

    def scannerA4(self):
        App.scanner(SIZE_S.A4)

    def scannerF4(self):
        App.scanner(SIZE_S.F4)

    def scannerCard(self):
        App.scanner(SIZE_S.CARD)

    def signalHandler(self, signal, frame):
        INFO("Catching SIGINT signal!")
        self.exit()

    def run(self, mousecb):
        signal.signal(signal.SIGINT, self.signalHandler)
        self.win = tk.Tk()
        self.win.title('Homography')
        self.win.geometry('300x50')
        
        self.menubar = tk.Menu(self.win)
        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label='File', menu=self.filemenu)

        self.filemenu.add_command(label='Open', command=self.open)
        self.filemenu.add_command(label='Save', command=self.save)
        self.filemenu.add_separator()
        self.filemenu.add_command(label='Exit', command=self.exit)

        self.modemenu = tk.Menu(self.menubar, tearoff=0)
        self.scannermenu = tk.Menu(self.modemenu, tearoff=0)
        self.menubar.add_cascade(label='Mode', menu=self.modemenu)
        self.modemenu.add_command(label='Panorama', command=self.panorama)
        self.modemenu.add_cascade(label='Scanner', menu=self.scannermenu)
        self.scannermenu.add_command(label='A4', command=self.scannerA4)
        self.scannermenu.add_command(label='F4', command=self.scannerF4)
        self.scannermenu.add_command(label='CARD', command=self.scannerCard)

        self.label = tk.Label(self.win, text='Select 4 corners', bg='green',
            font=('Arial', 10), width=13, height=2)
        self.btn_reset = tk.Button(self.win, text='reset',
            font=('Arial', 12), width=10, height=2,command=mousecb.play_reset)
        self.btn_apply = tk.Button(self.win, text='apply',
        font=('Arial', 12), width=10, height=2,command=mousecb.play_apply)

        self.win.config(menu=self.menubar)
        self.label.pack(fill=tk.Y, side=tk.LEFT)
        self.btn_reset.pack(fill=tk.Y, side=tk.LEFT)
        self.btn_apply.pack(fill=tk.Y, side=tk.LEFT)

        self.win.resizable(0,0)
        INFO('App Running!')
        self.win.mainloop()

class SIZE(object):
    __slots__ = ('w','h')

    def __init__(self):
        self.w = self.h = 1

    def set(self, sz):
        self.w = sz[0]
        self.h = sz[1]

class MouseCB(object):
    def __init__(self):
        self.cur_time = 0
        self.pt0 = [10,10]
        self.pt1 = [10,10]
        self.pt2 = [10,10]
        self.pt3 = [10,10]
        self.pt4 = [0,0]
        self.cursor = None
        self.flags = 0
        self.param = 0
        self.quit = 0
        self.cvtapp = None
        self.cvt_app = None
        self.cvt_up = 0
        self.cvt_sz = SIZE()
        self.cvt_sz_old = SIZE()

    def setSize(self, sz):
        if self.cvt_sz.w != sz[0] or self.cvt_sz.h != sz[1]:
            self.pt0 = self.convert(sz, self.pt0)
            self.pt1 = self.convert(sz, self.pt1)
            self.pt2 = self.convert(sz, self.pt2)
            self.pt3 = self.convert(sz, self.pt3)
            self.cvt_sz_old = self.cvt_sz
            self.cvt_sz.set(sz)

    def convert(self, imgsz, pts):
        return [int(pts[0]/self.cvt_sz.w*imgsz[0]), int(pts[1]/self.cvt_sz.h*imgsz[1])]

    def getPtsList(self):
        return [self.pt0, self.pt1, self.pt2, self.pt3]

    def getPtsListConvert(self, imgsz):
        k = [self.pt0, self.pt1, self.pt2, self.pt3]
        return [self.convert(imgsz,_k) for _k in k]

    def play_cb(self, event, x, y, flags, param):
        self.cursor = [x, y]
        if event == cv2.EVENT_LBUTTONDOWN:
            self.cur_time += 1
            if self.cur_time == 1:
                self.pt0 = [x, y]
            if self.cur_time == 2:
                self.pt1 = [x, y]
            if self.cur_time == 3:
                self.pt2 = [x, y]
            if self.cur_time == 4:
                self.pt3 = [x, y]
            if self.cur_time == TIME_MAX:
                self.cur_time = 4
        if event == cv2.EVENT_RBUTTONDOWN:
            self.cur_time = 0

    def getU(self):
        return np.array([[self.pt0[0], self.pt1[0], self.pt2[0], self.pt3[0]],
                         [self.pt0[1], self.pt1[1], self.pt2[1], self.pt3[1]],
                         [1, 1, 1, 1.0]])
    def getResizeU(self):
        global g_ctx
        h, w, c = g_ctx.img.shape
        u = self.getU()
        u[0,:] = u[0,:] / self.cvt_sz.w * w
        u[1,:] = u[1,:] / self.cvt_sz.h * h
        return u

    def play_reset(self):
        self.cur_time = 0
        INFOH('hit reset!')

    def play_apply_scanner(self):
        global g_ctx
        if (self.cur_time == 4 and g_ctx is not None):
            self.flags = 1
            u = self.getResizeU()
            box = [SCANNERV[1, 2], SCANNERV[0, 2]]
            g_ctx.oimg = transformImage(g_ctx.img, u, SCANNERV, box=box)
            self.startCvApp(CvTApp)
        else:
            INFO('hit apply! Please select region first')

    def play_apply_pano(self):
        global g_ctx
        if g_ctx.img is not None and g_ctx.img2 is not None:
            g_ctx.oimg = stitching(g_ctx.img2, g_ctx.img, 
                blending="Rate", th=4, 
                blendrate=0.2, d=95, k=1500, 
                override=0)
            self.startCvApp(CvPApp)
        else:
            INFO('hit apply! Please select region first')

    def play_apply(self):
        global g_ctx
        if g_ctx.mode == Control.SCANNER:
            self.play_apply_scanner()
        elif g_ctx.mode == Control.PANORAMA:
            self.play_apply_pano()

    def startCvApp(self, cvapp):
        self.cvtapp = cvapp()
        self.cvt_app = pthread.Thread(target=self.cvtapp.run)
        self.cvt_app.start()
        self.cvt_up = 1

    def exitCvTApp(self):
        global g_ctx
        g_ctx.cvt_quit = 1
        self.cvt_app.join()
        self.cvt_app = None
        self.cvt_up = 0
        g_ctx.cvt_quit = 0

class CvTApp(object):
    def run(self):
        cv2.namedWindow(WINT, cv2.WINDOW_NORMAL)
        while (g_ctx.oimg is None):
             time.sleep(0.1)
        img = g_ctx.oimg.copy()
        hh, ww, cc = img.shape
        cv2.resizeWindow(WINT, ww, hh)
        while (g_ctx.cvt_quit == 0):
            if (cv2.getWindowProperty(WINT, 0) < 0):
                 break
            show_img, sz = run_resize_image(img, WINT, ww, hh)
            cv2.imshow(WINT, show_img)
            key = cv2.waitKey(30)
            if (key == ord('q')):
                 break
        cv2.destroyWindow(WINT)

class CvPApp(object):
    def run(self):
        blank = np.zeros((H,W,C),dtype=np.uint8)
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        i = 0
        global g_ctx
        while (g_ctx.img_name == None):
            time.sleep(0.1)
        blank = g_ctx.oimg.copy()
        hh, ww, cc = blank.shape
        cv2.resizeWindow(WIN, ww, hh)
        while (g_ctx.cv_quit == 0):
            if (cv2.getWindowProperty(WIN, 0) < 0):
                break
            blk = blank.copy()
            show_img, sz = run_resize_image(blk, WIN, ww, hh)
            cv2.imshow(WIN, show_img)
            key = cv2.waitKey(30)
            if (key == ord('q')):
                break
        cv2.destroyAllWindows()

class CvApp(object):
    def __init__(self):
        pass

    def run(self, cb):
        blank = np.zeros((H,W,C),dtype=np.uint8)
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WIN, cb.play_cb)
        i = 0
        global g_ctx
        while (g_ctx.img_name == None):
            time.sleep(0.1)
        blank = g_ctx.img.copy()
        hh, ww, cc = blank.shape
        cv2.resizeWindow(WIN, ww, hh)
        while (g_ctx.cv_quit == 0):
            if (cv2.getWindowProperty(WIN, 0) < 0):
                break
            blk = blank.copy()
            i += 1
            #cv2.putText(blk, "%d" % i, 0.6, 1, 4, 1)
            #print(i, cb.cursor, cb.pt0, cb.pt1, cb.pt2, cb.pt3)
            blk, sz = run_resize_image(blk, WIN, ww, hh, lock=1)
            if cb.cur_time > 0:
                k = cb.getPtsList()
                if cb.cur_time < 4:
                    k[cb.cur_time] = cb.cursor
                    for l in range(cb.cur_time):
                        #print(k[l], k[l+1])
                        cv2.line(blk, tuple(k[l]), tuple(k[l+1]), (0,170,0), 3)
                if cb.cur_time == 4:
                    cv2.polylines(blk, np.array([k]) , 1, [0, 170,255], 10)
            _, _, sw, sh = cv2.getWindowImageRect(WIN)
            cb.setSize([sw,sh])
            cv2.imshow(WIN, blk)
            key = cv2.waitKey(30)
            if (key == ord('q')):
                break
        cv2.destroyAllWindows()

SCANNERV = np.array([[0, 400, 400., 0],
              [0, 0,   780., 780],
              [1,  1,  1,  1]])
g_ctx = Control()
App.scanner(SIZE_S.A4)
cb = MouseCB()

def run():
    global g_ctx, cb
    ctrl = App()
    ctrl.run(cb)
    g_ctx.cv_quit = 1
    g_ctx.cvt_quit = 1
    time.sleep(0.2)
    exit(0)

def main():
    run()

if __name__ == '__main__':
    main()
