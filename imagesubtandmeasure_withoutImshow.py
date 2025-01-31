# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:58:05 2017
parallel and staggered roi


Copyright (c) 2018, Taizo kawano

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
###########################################################################

1 Run this script
2 Click the open button and select image files directory
3 Choose a roi col x row from drop down list or manually
4 Adjust xy position and interval of roi by sliders
5 Click start and wait for about 40 min
6 Click save to save csv file in the image directory.
the csv file will be used later processing part.

210720 option for normalize images before subtraction. 
in case only a few animals imaged and no motion cause high bg.
but this way may not good for high bg images eg. empty cell, many bubbles...

230523 Fixed to support python 3.11.
by Shinichi Miyazaki

240507 Fixed to support python 3.12 and MacOS.
Work on following conditions:
- Python 3.12.0 (venv) on MacOS
- Python 3.9.5 (conda) on MacOS
"""

import os
import sys
import time
import datetime
import threading
import re
import csv
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tkinter
import tkinter.ttk
import tkinter.filedialog
import cv2
from tqdm import tqdm


############################################################
def main(args):
    print("run on console or as script, not as imported library")

    # test(args)
    # windowを表示している
    root = tkinter.Tk()
    root.title(u"Image processing")
    root.geometry("600x400")

    app = Imageprocessgui(master=root)
    # to test
    # app.chooseadir(imagedir = "C:/Users/Hayashi_Lab/Documents/imagedata/171218zxis22red5_10_20_50_1spm/jpgfiles0")
    # app.openadir()

    app.mainloop()

    # sys.exit()
    # return app


def test(args):
    print("test function")
    root = tkinter.Tk()
    root.title(u"Image processing")
    root.geometry("600x400")
    # root.protocol("WM_DELETE_WINDOW", on_closing)

    app = Imageprocessgui(master=root)
    app.chooseadir(imagedir="C:/Users/Hayashi_Lab/Documents/imagedata/171218zxis22red5_10_20_50_1spm/jpgfiles0")
    app.openadir()
    app.mainloop()
    animagedir = "C:/Users/Hayashi_Lab/Documents/imagedata/171218zxis22red5_10_20_50_1spm/jpgfiles0/img_000000000_Default_000.jpg"
    temp = cv2.imread(animagedir)

    pass


def on_closing():
    print("cv2.destroyAllWindows")
    cv2.destroyAllWindows()


##################################################################
class Imageprocessgui(tkinter.Frame):  ##tkinter.Frameを継承
    # ui default values
    # roi width
    droiwidth = 100
    # roi height
    droiheight = 100
    # roi column num
    droicolnum = 8
    # roi row num
    droirownum = 6
    # roi top left x
    dtopleftx = 23
    # roi top left y
    dtoplefty = 49
    # roi interval x
    dintervalx = 123
    # roi interval y
    dintervaly = 123
    # roi rotate degree
    drotate = 0
    # step to process slice. normaly 1 which means process all
    slicestep = 1

    ##__init__はコンストラクタと呼ばれclassが呼び出されたとき最初に実行されるメソッド
    def __init__(self, master=None):
        tkinter.Frame.__init__(self, master)
        title = "image processor"
        top = self.winfo_toplevel()
        top.title(title)
        # need to destryo cv2 processes?
        # top.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.imagedir = None
        self.jpgfilenamelist = []
        self.ims = Imagestack()

        # self.clearvar()

        # gui
        # open choose a directory contains images
        self.openbutton = tkinter.Button(text=u"open",
                                         width=10, height=3)
        self.openbutton.place(x=10, y=10)
        self.orignalbuttoncolor_o = self.openbutton.cget("background")
        # <Button-1> left mous button pushed.
        # ButtonRelease is better
        self.openbutton.bind("<ButtonRelease-1>", self.setadir)

        # open subtracted windwo
        self.subtbutton = tkinter.Button(text=u"processwindow",
                                         width=10, height=3)
        self.subtbutton.place(x=10, y=220)
        self.orignalbuttoncolor_o = self.subtbutton.cget("background")
        # <Button-1> left mous button pushed.
        # ButtonRelease is better
        self.subtbutton.bind("<ButtonRelease-1>", self.showprocesswindow)

        # start processing
        self.startbutton = tkinter.Button(text=u"start",
                                          width=10, height=3)
        self.startbutton.place(x=10, y=300)
        self.orignalbuttoncolor_o = self.startbutton.cget("background")
        # <Button-1> left mous button pushed.
        # ButtonRelease is better
        self.startbutton.bind("<ButtonRelease-1>", self.startprocess)  #
        # pass
        # save
        self.savebutton = tkinter.Button(text=u"save",
                                         width=10, height=3)
        self.savebutton.place(x=200, y=300)
        self.orignalbuttoncolor_o = self.savebutton.cget("background")
        # <Button-1> left mous button pushed.
        # ButtonRelease is better
        self.savebutton.bind("<ButtonRelease-1>", self.savedata)  #

        # temporal for debug
        self.tempbutton1 = tkinter.Button(text=u"temp1",
                                          width=10, height=3)
        self.tempbutton1.place(x=300, y=300)
        self.orignalbuttoncolor_o = self.tempbutton1.cget("background")
        # <Button-1> left mous button pushed.
        # ButtonRelease is better
        self.tempbutton1.bind("<ButtonRelease-1>", self.tempfunc1)  #

        # slice
        # start box
        self.boxstart = self.setabox(100, 300, text=u"start")
        # get the value
        self.slicestartstr = self.boxstart.get()

        self.boxend = self.setabox(100, 340, text=u"end")
        # get the value
        self.sliceendstr = self.boxend.get()

        # roi button
        self.roibutton = tkinter.Button(text=u"auto roi",
                                        width=10, height=3)
        self.roibutton.place(x=100, y=10)
        self.orignalbuttoncolor_o = self.roibutton.cget("background")
        # <Button-1> left mous button pushed.
        # ButtonRelease is better
        self.roibutton.bind("<ButtonRelease-1>", self.autosetroi)  #

        # roi: width, height, column num, row num, top left coordinate.
        # roi width
        self.boxwidth = self.setabox(200, 10,
                                     default=Imageprocessgui.droiwidth,
                                     text=u"width")
        self.roiwidthstr = self.boxwidth.get()
        # roi height
        self.boxheight = self.setabox(200, 50,
                                      default=Imageprocessgui.droiheight,
                                      text=u"height")
        self.roiheightstr = self.boxheight.get()
        # roi column num
        self.boxcolnum = self.setabox(300, 10,
                                      default=Imageprocessgui.droicolnum,
                                      text=u"column")
        self.roicolnumstr = self.boxcolnum.get()
        # roi row num
        self.boxrownum = self.setabox(300, 50,
                                      default=Imageprocessgui.droirownum,
                                      text=u"row")
        self.roirownumstr = self.boxrownum.get()

        # roi colxrow dropdown list
        self.crb = tkinter.ttk.Combobox(master, state="readonly")
        self.crb["values"] = ("12x8", "8x6", "6x4", "4x3", "2x2")
        self.crb.current(1)
        self.crb.place(x=400, y=10)
        self.crb.bind("<<ComboboxSelected>>", self.crb_selected)

        # normalize before subtract
        # check box
        self.nbs_check_val = tkinter.BooleanVar(value=False)
        self.nbs_check = tkinter.Checkbutton(master,
                                             text="pre normalize",
                                             # command = self.nbs_check_click(),
                                             variable=self.nbs_check_val
                                             )
        self.nbs_check.place(x=400, y=50)

        # roi top left x
        self.boxtopleftx = self.setabox(10, 100,
                                        default=Imageprocessgui.dtopleftx,
                                        text=u"x")
        self.roitopleftxstr = self.boxtopleftx.get()
        # roi top left y
        self.boxtoplefty = self.setabox(10, 140,
                                        default=Imageprocessgui.dtoplefty,
                                        text=u"y")
        self.roitopleftystr = self.boxtoplefty.get()
        # roi interval x
        self.boxintervalx = self.setabox(300, 100,
                                         default=Imageprocessgui.dintervalx,
                                         text=u"x interval")
        self.roiintervalxstr = self.boxintervalx.get()
        # roi interval y
        self.boxintervaly = self.setabox(300, 140,
                                         default=Imageprocessgui.dintervaly,
                                         text=u"y interval")
        self.roiintervalystr = self.boxintervaly.get()
        # roi rotate degree
        self.boxrotate = self.setabox(200, 180,
                                      default=Imageprocessgui.drotate,
                                      text=u"rotate")
        self.roirotatestr = self.boxrotate.get()

        # slider for top left x
        self.sliderx = tkinter.IntVar()
        self.sliderx.set(Imageprocessgui.dtopleftx)
        # scale
        s1 = tkinter.Scale(master, label="", orient='h', \
                           length=180, from_=0, to=2047, \
                           variable=self.sliderx)
        """,\
                           command = self.changex)"""
        # s1.bind("<ButtonRelease-1>", 
        s1.bind("<B1-Motion>",
                lambda event: self.sliderchanged(event, "s1"))
        s1.bind("<ButtonRelease-1>",
                lambda event: self.sliderchanged(event, "s1"))
        s1.place(x=100, y=80)

        # slider for top left y
        self.slidery = tkinter.IntVar()
        self.slidery.set(Imageprocessgui.dtoplefty)
        # scale
        s2 = tkinter.Scale(master, label="", orient='h', \
                           length=180, from_=0, to=2047, \
                           variable=self.slidery)
        """,\
                           command = self.changex)"""
        s2.bind("<B1-Motion>",
                lambda event: self.sliderchanged(event, "s2"))
        s2.bind("<ButtonRelease-1>",
                lambda event: self.sliderchanged(event, "s2"))
        s2.place(x=100, y=120)

        # slider for interval x
        self.sliderintx = tkinter.IntVar()
        self.sliderintx.set(Imageprocessgui.dintervalx)
        # scale
        s3 = tkinter.Scale(master, label="", orient='h', \
                           length=180, from_=0, to=512, \
                           variable=self.sliderintx)
        """,\
                           command = self.changex)"""
        # s1.bind("<ButtonRelease-1>", 
        s3.bind("<B1-Motion>",
                lambda event: self.sliderchanged(event, "s3"))
        s3.bind("<ButtonRelease-1>",
                lambda event: self.sliderchanged(event, "s3"))
        s3.place(x=400, y=80)

        # slider for interval y
        self.sliderinty = tkinter.IntVar()
        self.sliderinty.set(Imageprocessgui.dintervaly)
        # scale
        s4 = tkinter.Scale(master, label="", orient='h', \
                           length=180, from_=0, to=512, \
                           variable=self.sliderinty)
        """,\
                           command = self.changex)"""
        s4.bind("<B1-Motion>",
                lambda event: self.sliderchanged(event, "s4"))
        s4.bind("<ButtonRelease-1>",
                lambda event: self.sliderchanged(event, "s4"))
        s4.place(x=400, y=120)

        # threshold of subtracted image as SD
        self.boxthreshold = self.setabox(10, 200, default=2, text=u"threshold")
        self.thresholdstr = self.boxthreshold.get()

        # step. 1 is every slice
        self.boxslicestep = self.setabox(180, 200, default=1, text=u"step")
        self.slicestepstr = self.boxslicestep.get()

    """   
    def on_closing(self,dummy_arg=None):
        print("cv2.destroyAllWindows")
        #cv2.destroyAllWindows()
        #self.quit()
    """

    def nbs_check_click(self, event):
        pass

    def crb_selected(self, event):
        crbstr = self.crb.get()
        print("col row val " + crbstr)
        coln = self.boxcolnum.get()
        rown = self.boxrownum.get()
        if crbstr == "12x8":
            coln = str(12)
            rown = str(8)
        elif crbstr == "8x6":
            coln = str(8)
            rown = str(6)
        elif crbstr == "6x4":
            coln = str(6)
            rown = str(4)
        elif crbstr == "4x3":
            coln = str(4)
            rown = str(3)
        elif crbstr == "2x2":
            coln = str(2)
            rown = str(2)

        self.boxcolnum.delete(0, tkinter.END)
        self.boxcolnum.insert(tkinter.END, str(coln))
        self.boxrownum.delete(0, tkinter.END)
        self.boxrownum.insert(tkinter.END, str(rown))

        # adjust by image width, height
        if self.ims is not None:
            tempimage = self.ims.getaimage(0)
            imageshape = tempimage.shape
            iwidth = imageshape[1]
            iheight = imageshape[0]
            xinterval = iwidth / (float(self.boxcolnum.get()) + 0.36)
            self.boxintervalx.delete(0, tkinter.END)
            self.boxintervalx.insert(tkinter.END, str(xinterval))
            self.sliderintx.set(xinterval)
            yinterval = iheight / (float(self.boxrownum.get()) + 0.36)
            self.boxintervaly.delete(0, tkinter.END)
            self.boxintervaly.insert(tkinter.END, str(yinterval))
            self.sliderinty.set(yinterval)

            self.setroi(event)

    def sliderchanged(self, event, slidername):

        # print(dir(event.widget))
        # print(event.widget.cget("label"))
        # print(slidername)
        if slidername == "s1":
            value = self.sliderx.get()
            # print(value)
            self.boxtopleftx.delete(0, tkinter.END)
            self.boxtopleftx.insert(tkinter.END, str(value))
        elif slidername == "s2":
            value = self.slidery.get()
            # print(value)
            self.boxtoplefty.delete(0, tkinter.END)
            self.boxtoplefty.insert(tkinter.END, str(value))
        elif slidername == "s3":
            value = self.sliderintx.get()
            # print(value)
            self.boxintervalx.delete(0, tkinter.END)
            self.boxintervalx.insert(tkinter.END, str(value))
        elif slidername == "s4":
            value = self.sliderinty.get()
            # print(value)
            self.boxintervaly.delete(0, tkinter.END)
            self.boxintervaly.insert(tkinter.END, str(value))

        self.setroi(event)

    def setabox(self, x, y, **kwargs):
        if "text" in kwargs:
            text = kwargs["text"]
        if "width" in kwargs:
            width = kwargs["width"]
        else:
            width = 6
        if "default" in kwargs:
            default = kwargs["default"]
        else:
            default = 0
        # label
        label = tkinter.Label(text=text)
        label.place(x=x, y=y)
        boxarg = tkinter.Entry(width=width)
        boxarg.insert(tkinter.END, str(default))
        boxarg.place(x=x + 50, y=y)
        return boxarg
        # get the value
        # self.roiwidthstr = boxarg.get() 
        """
        #roinum box
        self.boxwidth = tkinter.Entry(width = width )
        self.boxwidth.insert(tkinter.END, str(default))
        self.boxwidth.place(x=x+50, y = y)
        # get the value
        self.roiwidthstr = self.boxwidth.get() 
        """

    # clear instance var here
    def clearvar(self):
        # instance var
        self.imagedir = None
        self.jpgfilenamelist = []
        self.ims = None
        return

    def setadir(self, event):
        self.chooseadir()
        self.openadir()

    def chooseadir(self, **kwargs):
        if "imagedir" in kwargs:
            self.imagedir = kwargs["imagedir"]
        else:
            tk = tkinter.Tk()
            tk.withdraw()
            # directory that contains csv files
            self.imagedir = tkinter.filedialog.askdirectory()
            # this line needed to propary return to console.
            tk.destroy()
        print("self.imagedir " + self.imagedir)

    def openadir(self):
        """
        tk = tkinter.Tk()
        tk.withdraw()
        # directory that contains csv files
        self.imagedir = tkinter.filedialog.askdirectory()
        #this line needed to propary return to console.
        tk.destroy()
        print("self.imagedir "+self.imagedir)
        """
        # return 
        dircontents = os.listdir(self.imagedir)
        # print("dircontents " + str(dircontents))
        self.jpgfilenamelist = []
        for x in dircontents:
            # need separater
            # if os.path.isfile(targetdir + x):
            if os.path.isfile(os.path.join(self.imagedir, x)):
                # print(str(x))
                if ".jpg" in x:
                    self.jpgfilenamelist.append(x)
        if len(self.jpgfilenamelist) > 0:
            print("len(self.jpgfilenamelist) " + str(len(self.jpgfilenamelist)))
            self.ims.setdir(self.imagedir, self.jpgfilenamelist)
            self.sliceendstr = str(len(self.jpgfilenamelist) - 1)
            self.boxend.delete(0, tkinter.END)
            self.boxend.insert(tkinter.END, self.sliceendstr)

        else:
            print("The directory doesnt have jpg")
        return

    def showprocesswindow(self, event):
        self.slicestartstr = self.boxstart.get()
        self.sliceendstr = self.boxend.get()
        self.startslice = int(self.slicestartstr)
        self.endslice = int(self.sliceendstr)
        print(str(self.startslice) + ":" + str(self.endslice))
        self.threshold = float(self.boxthreshold.get())
        self.slicestep = int(self.boxslicestep.get())

        self.subtwindowname = "subtmed"
        cv2.startWindowThread()
        cv2.namedWindow(self.subtwindowname, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('slice', self.subtwindowname,
                           0, self.endslice, self.showsubtmedimg)
        self.ip = Imageprocess(self, windowname=self.subtwindowname,
                               threshold=self.threshold,
                               slicestep=self.slicestep)

    def startprocess(self, event):
        print("start ")
        """
        self.slicestartstr = self.boxstart.get() 
        self.sliceendstr = self.boxend.get() 
        self.startslice = int(self.slicestartstr)
        self.endslice = int(self.sliceendstr)
        print(str(self.startslice)+":"+str(self.endslice))
        self.threshold = float(self.boxthreshold.get())

        self.subtwindowname = "subtmed"
        cv2.startWindowThread()
        cv2.namedWindow(self.subtwindowname, cv2.WINDOW_NORMAL)
        cv2.createTrackbar('slice',self.subtwindowname,
                              0,self.endslice, self.showsubtmedimg)
        self.ip = Imageprocess(self, windowname = self.subtwindowname,
                               threshold = self.threshold)
        #self.ip = Imageprocess(self, save = True)
        """
        self.showprocesswindow(event)
        self.outputdata = None
        self.ip.start()
        # print(len(self.outputdata))
        # self.ip.join()

    def savedata(self, event, **kwargs):
        if self.outputdata is not None:
            dataframe = pandas.DataFrame(self.outputdata,
                                         columns=["Area" for a in range(self.roicol.getlen())])
            dataframe.to_csv(os.path.join(self.imagedir, "area.csv"), index=False)
        return

    def tempfunc1(self, event, **kwargs):

        if not self.ip.is_alive():
            # print("ip is Not alive")
            self.threshold = float(self.boxthreshold.get())
            # print(n)
            # print(self.threshold)
            # self.sutmedimageslice = n
            n = self.sutmedimageslice
            """
            img1 = self.ims.getaimage(n).astype(np.float32)
            filename="img1.tif"
            filepass = os.path.join(self.imagedir, filename)
            cv2.imwrite(filepass, img1)
            return
            """

            if n < self.ims.nslice - 1:
                subtractor = Subtractor(self)
                img1 = self.ims.getaimage(n).astype(np.float32)
                img2 = self.ims.getaimage(n + 1).astype(np.float32)
                subimage = subtractor.subtract(img1, img2)
                # 180515
                # this median blur cause large difference from imagej?
                # subimage is almost same.
                subtmedimg = cv2.medianBlur(subimage, 5)
                # print(subtmedimg.shape)#(768,1024,3)
                # olimg = self.overlaythreshold(subtmedimg, self.threshold)
                # cv2.imshow(self.subtwindowname, subtmedimg)
                # cv2.imshow(self.subtwindowname, olimg)

            # filename="".join([str(num),'.tif'])
            filename = "subtmed.tif"
            # tif format saving seems not work in this way. need fix. it is saved as binary (black/white)image?
            # quality 80? 95? imagej default seems 85
            filepass = os.path.join(self.imagedir, filename)
            """
            cv2.imwrite(filepass, img,
                        [int(cv2.IMWRITE_JPEG_QUALITY),85])
            """
            cv2.imwrite(filepass, subtmedimg)
            msgstr = "saved in " + str(filepass)
            print(msgstr)
            """
            filename="img1.tif"
            filepass = os.path.join(self.imagedir, filename)
            cv2.imwrite(filepass, img1)
            filename="img2.tif"
            filepass = os.path.join(self.imagedir, filename)
            cv2.imwrite(filepass, img2)
            """
            filename = "subimage.tif"
            filepass = os.path.join(self.imagedir, filename)
            cv2.imwrite(filepass, subimage)

        return

    def setroi(self, event):
        self.roicol = Roicollection(self)
        # self.roilist = []
        # roi column num
        roicolnum = int(self.boxcolnum.get())
        # roi row num
        roirownum = int(self.boxrownum.get())
        # roi interval x
        roiintervalx = float(self.boxintervalx.get())
        # roi interval y
        roiintervaly = float(self.boxintervaly.get())
        # topleft
        x = int(self.boxtopleftx.get())
        y = int(self.boxtoplefty.get())
        width = int(self.boxwidth.get())
        height = int(self.boxheight.get())
        radianrot = np.pi * float(self.boxrotate.get()) / 180

        roisarg = [roicolnum, roirownum, roiintervalx, roiintervaly,
                   x, y, width, height, radianrot]
        self.roicol.setrois(*roisarg)
        self.ims.showrois(self.roicol)
        """
        slicepos = self.ims.slicepos
        baseimage = self.ims.getaimage(slicepos)
        self.roicol.showrois(baseimage, "image")
        """

    def autosetroi(self, event):
        sys.path.append("C:\\Users\\Hayashi_Lab\\Documents\\programs\\imagecapture")

        import autocircledetect
        self.roicol = Roicollection(self)

        awd = autocircledetect.Autowelldetector()
        awd.setimg(self.ims.getaimage(0))

        awd.searchcircles()
        # awd.detectcircles()
        # awd.showimg(awd.img)
        awd.drawcircles()
        awd.filterbyradiusdistribution()
        colpos, rowpos, roiwidth = awd.estimatelattice()
        # awd.showimg("detected circles",awd.cimg)

        self.roicol.autosetrois(colpos, rowpos, roiwidth)
        self.ims.showrois(self.roicol)

        # set values in gui box
        # top left corner
        topleftx = int(colpos[0] - roiwidth / 2)
        self.boxtopleftx.delete(0, tkinter.END)
        self.boxtopleftx.insert(tkinter.END, str(int(topleftx)))
        self.sliderx.set(topleftx)

        toplefty = int(rowpos[0] - roiwidth / 2)
        self.boxtoplefty.delete(0, tkinter.END)
        self.boxtoplefty.insert(tkinter.END, str(int(toplefty)))
        self.slidery.set(toplefty)

        # col row numbers
        self.boxcolnum.delete(0, tkinter.END)
        self.boxcolnum.insert(tkinter.END, str(len(colpos)))
        self.boxrownum.delete(0, tkinter.END)
        self.boxrownum.insert(tkinter.END, str(len(rowpos)))

        # intervals
        xinterval = np.mean(np.diff(colpos))
        self.boxintervalx.delete(0, tkinter.END)
        self.boxintervalx.insert(tkinter.END, str(int(xinterval)))
        self.sliderintx.set(xinterval)
        yinterval = np.mean(np.diff(rowpos))
        self.boxintervaly.delete(0, tkinter.END)
        self.boxintervaly.insert(tkinter.END, str(int(yinterval)))
        self.sliderinty.set(yinterval)

        # roi width
        self.boxwidth.delete(0, tkinter.END)
        self.boxwidth.insert(tkinter.END, str(roiwidth))
        self.boxheight.delete(0, tkinter.END)
        self.boxheight.insert(tkinter.END, str(roiwidth))

    def showsubtmedimg(self, n):
        # if self.ip.is_alive():
        #    print("ip is alive")
        # """
        if not self.ip.is_alive():
            # print("ip is Not alive")
            self.threshold = float(self.boxthreshold.get())
            self.sutmedimageslice = n
            # print(n)
            # print(self.threshold)
            if n < self.ims.nslice - 1:
                subtractor = Subtractor(self)
                img1 = self.ims.getaimage(n).astype(np.float32)
                img2 = self.ims.getaimage(n + 1).astype(np.float32)
                subimage = subtractor.subtract(img1, img2)
                subtmedimg = cv2.medianBlur(subimage, 5)
                # print(subtmedimg.shape)#(768,1024,3)
                olimg = self.overlaythreshold(subtmedimg, self.threshold)
                # cv2.imshow(self.subtwindowname, subtmedimg)
                cv2.imshow(self.subtwindowname, olimg)
                # cv2.setTrackbarPos('slice',self.windowname,i)
        # """
        pass

    def overlaythreshold(self, img, val):
        # colimg = np.zeros(img.shape, dtype = np.uint8)
        threshold = 127 - val * 12.8
        # print(threshold)
        colimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # retval, binaryimg = cv2.threshold(img[:,:,2],threshold,1,
        #                                   cv2.THRESH_BINARY_INV)
        retval, binaryimg = cv2.threshold(colimg[:, :, 2], threshold, 1,
                                          cv2.THRESH_BINARY_INV)
        # mask = cv2.inRange(img, np.array([0,0,0]),np.array([val,val,val]))
        # red channel =2, green =1, blue = 0
        # colimg[:,:,2][binaryimg == 1] = 255
        # img[:,:,2][binaryimg == 1] = 255
        colimg[:, :, 2][binaryimg == 1] = 255
        # colimg[:,:,2][img[:,:,2] < threshold] = 255
        # colimg[img < val] = [[[0,0,255]]]
        # return colimg + img
        return colimg
        pass

    def toggle_l(self, event):
        # self.writeinlog("toggle1")
        # global cause error only made a exe with cx_freeze?
        # global onofftoggle1
        # self.writeinlog(str(self.onofftoggle1))
        if self.startstoptoggle_l:
            self.button_l.config(text=u"Stop live", bg="red")
            self.startstoptoggle_l = False
            self.livestart()
        else:
            self.button_l.config(text=u"Live", bg=self.orignalbuttoncolor_l)
            self.startstoptoggle_l = True
            self.livestop()


##################################################################
class Imageprocess(threading.Thread):
    # , startslice, endslice, ipg, saveflag = False
    def __init__(self, ipg, **kwargs):
        super(Imageprocess, self).__init__()
        self.ipg = ipg
        self.saveflag = False
        self.threshold = 3
        self.slicestep = 1
        self.windowname = "subtmed"
        # self.ipg.subtwindowname = self.windowname
        if "threshold" in kwargs:
            self.threshold = kwargs["threshold"]
        if "slicestep" in kwargs:
            self.slicestep = kwargs["slicestep"]
        if "save" in kwargs:
            self.saveflag = True

        if "windowname" in kwargs:
            self.windowname = kwargs["windowname"]
        else:
            cv2.startWindowThread()
            cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
            cv2.createTrackbar('slice', self.windowname,
                               0, int(self.ipg.endslice / self.slicestep),
                               self.nothing)

    def nothing(self, n):
        pass

    def run(self):
        print("start at " + "_".join([datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')]))
        self.image1 = None
        self.startslice = self.ipg.startslice
        self.endtslice = self.ipg.endslice
        self.processnum = int((self.endtslice - self.startslice) / (self.slicestep))
        self.output = np.zeros((self.processnum,
                                self.ipg.roicol.getlen()))
        subtractor = Subtractor(self.ipg)
        subtractor.setinitialimage(self.ipg.ims.getaimage(self.startslice))

        for i in tqdm(range(self.processnum)):
            currentslice = int(self.startslice + i * self.slicestep)
            nextimg = self.ipg.ims.getaimage(currentslice + int(1 * self.slicestep))
            subimage = subtractor.subtractfromholdingimage(nextimg)
            subtractor.set2ndimageas1st()
            # on imagej macro run("Median...", "radius=2");
            # it may coresspond with 3? no. it seems imagejs radian =1
            # so need 6?odd5 or rgb 9?
            # 5 seems closer to imagej radian=2. looks differennt algorism.
            # so may not able to give same result.
            subtmedimg = cv2.medianBlur(subimage, 5)

            # subimage = self.subtract(currentslice,currentslice+1) 
            if self.saveflag == True:
                self.saveaimage(subtmedimg, i)
            # cv2.imshow(self.windowname, subtmedimg)
            # cv2.setTrackbarPos('slice',self.windowname,i)

            # 3sd 127-3*12.7
            threshold = 127 - self.threshold * 12.8
            retval, binaryimg = cv2.threshold(subtmedimg, threshold, 1,
                                              cv2.THRESH_BINARY_INV)
            # here must have roi processing part
            areadata = self.ipg.roicol.measureareas(binaryimg)
            self.output[i, :] = areadata

        print("end at " + "_".join([datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')]))
        self.ipg.outputdata = self.output
        # return self.output

    def saveaimage(self, img, num):
        # filename="".join([datetime.datetime.now().strftime('%Y%m%d%H%M%S_%f'),'.jpg'])
        # filename="".join([str(num),'.jpg'])
        filename = "".join([str(num), '.tif'])
        # tif format saving seems not work in this way. need fix. it is saved as binary (black/white)image?
        # quality 80? 95? imagej default seems 85
        filepass = os.path.join(self.ipg.imagedir, filename)
        """
        cv2.imwrite(filepass, img,
                    [int(cv2.IMWRITE_JPEG_QUALITY),85])
        """
        cv2.imwrite(filepass, img)
        msgstr = "saved in " + str(filepass)
        print(msgstr)


##################################################################
# 201706 modify normalization. before subtract normalize?
class Subtractor():
    # def __init__self(self):
    def __init__(self, ipg):
        self.ipg = ipg
        pass

    def setinitialimage(self, img):
        # convert from uint8 to float32             
        imgf32 = img.astype(np.float32)
        self.image1f32 = imgf32

    def set2ndimageas1st(self):
        self.image1f32 = self.image2f32

    def subtractfromholdingimage(self, img, **kwargs):
        imgf32 = img.astype(np.float32)
        self.image2f32 = imgf32
        return self.subtract(self.image1f32, self.image2f32, **kwargs)

    def subtract(self, img1, img2, **kwargs):

        # range to keep in 
        sdrange = 10
        if "sdrange" in kwargs:
            sdrange = kwargs["sdrange"]
        # gui pre normalize checkbox self.nbs_check.get()
        nbs_check_flag = self.ipg.nbs_check_val.get()
        # print("nbs " +str(nbs_check_flag))
        if nbs_check_flag:
            # 210719 normalize before subt. sd =1
            imgmean1 = np.mean(img1)
            imgsd1 = np.std(img1)
            normimg1 = (img1 - imgmean1) / imgsd1
            imgmean2 = np.mean(img2)
            imgsd2 = np.std(img2)
            normimg2 = (img2 - imgmean2) / imgsd2
            subtimgf32 = normimg1 - normimg2
            # normalized image sd = 1   2.5 seems similar
            subtimg = self.convertfloatTo8bit(subtimgf32, -2.5, 2.5)
        else:
            subtimgf32 = img1 - img2
            sd = np.std(subtimgf32)
            subtimg = self.convertfloatTo8bit(subtimgf32 - np.mean(subtimgf32),
                                              sd * sdrange * -1, sd * sdrange)

        return subtimg

    # convert flot array to 8bit array with vmin as0, vmax as255
    def convertfloatTo8bit(self, array, vmin, vmax):
        # temparray = array.astype(dtype).copy()
        temparray = array.copy()
        temparray = temparray - vmin
        temparray = temparray / (vmax - vmin) * 255
        # temparray[temparray<vmin] = 0
        # temparray[temparray>vmax] = 255
        temparray[temparray < 0] = 0
        temparray[temparray > 255] = 255
        return temparray.astype(np.uint8)


##################################################################
class Imagestack():

    def __init__(self):
        self.windowname = "image"
        pass

    def setdir(self, imagedir, imagenamelist):
        print("setdir")
        self.imagedir = imagedir
        self.imagenamelist = imagenamelist
        self.nslice = len(self.imagenamelist)
        self.slicepos = 0
        self.cr = Contrast()
        cv2.startWindowThread()
        # WINDOW_NORMAL  need to make flexible size window
        # cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.windowname,
                        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.createTrackbar('slice', self.windowname,
                           0, self.nslice - 1, self.readaimage)
        self.readaimage(0)
        self.cr.showhistogram(self.image)

    def clear(self):
        self.imagedir = None
        self.imagenamelist = []
        self.nslice = None
        cv2.destroyAllWindows()

    # the n start from 0. not 1
    def readaimage(self, n):
        # print("read image #"+str(n))
        imagepass = os.path.join(self.imagedir, self.imagenamelist[n])
        self.image = cv2.imread(imagepass)
        imageshape = self.image.shape
        cv2.resizeWindow(self.windowname, imageshape[1], imageshape[0])
        # equimg = cv2.equalizeHist(self.image)
        # cv2.imshow('image',equimg)
        if self.cr.adjusted:
            newimage = self.cr.changecontrast(self.image)
            cv2.imshow(self.windowname, newimage)
        else:
            cv2.imshow(self.windowname, self.image)
        self.slicepos = n

    def getaimage(self, n):
        # print("read image #"+str(n))
        imagepass = os.path.join(self.imagedir, self.imagenamelist[n])
        self.slicepos = n
        # return cv2.imread(imagepass)
        # 180427 if it is color, change to monochrome
        # actually always read as BGR?
        # rawimage = cv2.imread(imagepass)
        # cv2.IMREAD_GRAYSCALE
        rawimage = cv2.imread(imagepass, cv2.IMREAD_GRAYSCALE)
        """
        if len(rawimage.shape) == 3:
            gray = cv2.cvtColor(rawimage, cv2.COLOR_BGR2GRAY)
            outputimg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            outputimg = rawimage
        return outputimg
        """
        return rawimage

    def changeslice(self, n):
        # print(n)
        pass

    def showrois(self, roicollection):
        passimage = None
        if self.cr.adjusted:
            passimage = self.cr.changecontrast(self.image)
        else:
            passimage = self.image.copy()
        roicollection.showrois(passimage, self.windowname)
        pass


class Contrast():
    def __init__(self):
        self.imghight = 200
        self.imgwidth = 256
        self.min = 0
        self.max = 255
        self.adjusted = False
        self.windowname = "contrast"
        pass

    def autocont(self, image):
        return image
        pass

    def showhistogram(self, image):
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # do calc
        # tempimage = self.ic.image
        self.tempimage = image
        hist = cv2.calcHist([self.tempimage], [0], None, [256], [0, 256])
        histimagesorce = np.zeros((self.imghight, self.imgwidth, 3))
        for i, val in enumerate(hist):
            # value must be adjusted by image size (pixel num)
            # and max value of hist?
            cv2.line(histimagesorce, (int(i), int(0)), (int(i), int(val / 100)), (int(255), int(255), int(255)))
        self.histimage = np.flipud(histimagesorce)
        """cv2.putText(histimage, str(i+1)+"/"+str(self.maxframe), 
                            (50,50), font, 1, 255,2)
        """
        # this line is just work around for the opencv bug?
        # https://stackoverflow.com/questions/30249053/python-opencv-drawing-errors-after-manipulating-array-with-numpy
        """
        histimage = histimage.copy()
        cv2.putText(histimage, "median "+str(np.median(tempimage)),
                     (10,12), font, 0.5, (255,255,255),
                     1, cv2.LINE_AA)
        """
        cv2.imshow(self.windowname, self.histimage)

        cv2.createTrackbar('min', self.windowname,
                           0, 255, self.setmin)
        cv2.createTrackbar('max', self.windowname,
                           255, 255, self.setmax)
        if self.min != 0 or self.max != 255:
            self.drawaline()

    def setmin(self, v):
        self.adjusted = True
        # print("max min"+str(self.max)+" "+str(self.min)+ " v "+str(v))
        if self.max > v:
            self.min = v
            self.drawaline()
            newimage = self.changecontrast(self.tempimage)
            cv2.imshow('image', newimage)

    def setmax(self, v):
        self.adjusted = True
        # print("max min"+str(self.max)+" "+str(self.min)+ " v "+str(v))
        if v > self.min:
            self.max = v
            self.drawaline()
            newimage = self.changecontrast(self.tempimage)
            cv2.imshow('image', newimage)

    def drawaline(self):
        histimagewithline = self.histimage.copy()
        cv2.line(histimagewithline,
                 (self.min, 0), (self.min, self.imghight), (255, 0, 0))
        cv2.line(histimagewithline,
                 (self.max, 0), (self.max, self.imghight), (0, 0, 255))
        cv2.imshow(self.windowname, histimagewithline)

    def calclut(self):
        # make look up table with min max
        self.lut = np.arange(256, dtype=np.uint8)
        for i in range(0, self.min):
            self.lut[i] = 0
        for i in range(self.min, self.max):
            self.lut[i] = (i - self.min) / (self.max - self.min) * 255
        for i in range(self.max, 255):
            self.lut[i] = 255

    def changecontrast(self, img):
        if self.max > self.min:
            self.calclut()
            # overshoot: 255+10 = 9 
            # newimage = (img-self.min)/(self.max-self.min)*255
            # np.full(image.shape, val)
            # newimage = cv2.subtract(newimage, val)
            # newimage[newimage>255] = 255
            # newimage[newimage<0] = 0

            # use look up table 
            contrastimage = cv2.LUT(img, self.lut)
            return contrastimage

        # return newimage


"""
lut = np.arange(256, dtype = np.uint8)
for i in range (100, 150):
    lut[i] = (i-100)/(150 - 100)*255
"""


##################################################################
class Roi():

    def __init__(self, ipg):
        self.ipg = ipg

    def setname(self, name):
        self.name = name

    # top left position of roi
    def setpos(self, x, y):
        self.x = x
        self.y = y

    def setwidth(self, v):
        self.width = v

    def setheight(self, v):
        self.height = v

    # def show(self, slicepos):
    def show(self, image):
        modimage = image
        # 0,255,255 yellow
        cv2.rectangle(modimage,
                      (self.x, self.y),
                      (self.x + self.width, self.y + self.height),
                      (0, 255, 255),
                      2)
        return modimage
        # cv2.imshow("image", modimage)

    # the img must be binary. 0 or 1
    def measurearea(self, img):
        roiimg = img[self.y:self.y + self.height + 1, self.x:self.x + self.width + 1]
        value = np.sum(roiimg)
        return value

    ##################################################################


class Roicollection():
    # upperobj could be imageprocessgui, imagecapturegui?
    def __init__(self, upperobj):
        self.i = 0
        self.upperobj = upperobj
        self.roilist = []

    def __iter__(self):
        return iter(self.roilist)

    def next(self):

        if self.i == len(self.roilist):
            raise StopIteration
        val = self.roilist[self.i]
        self.i = self.i + 1
        return val

    def getlen(self):
        return len(self.roilist)

    """
    roisarg = [roicolnum, roirownum, roiintervalx, roiintervaly,
               x, y, width, height, radianrot]
    self.roilist.setrois(**roisarg)
    """

    def setrois(self, roicolnum, roirownum, roiintervalx, roiintervaly,
                x, y, width, height, radianrot):
        for row in range(roirownum):
            shifty = row * roiintervaly
            for col in range(roicolnum):
                aroi = Roi(self.upperobj)
                self.appendroi(aroi)
                shiftx = col * roiintervalx
                rotx = shiftx * np.cos(radianrot) - shifty * np.sin(radianrot)
                roty = shiftx * np.sin(radianrot) + shifty * np.cos(radianrot)
                xpos = int(x + rotx)
                ypos = int(y + roty)
                aroi.setpos(xpos, ypos)
                aroi.setheight(height)
                aroi.setwidth(width)
                # print("pos "+str(int(x+shiftx))+" "+str(int(y+shifty)))

    def autosetrois(self, colpos, rowpos, roiwidth):
        for i in range(len(rowpos)):
            for j in range(len(colpos)):
                aroi = Roi(self.upperobj)
                self.appendroi(aroi)
                aroi.setpos(int(colpos[j] - roiwidth / 2),
                            int(rowpos[i] - roiwidth / 2))
                aroi.setheight(int(roiwidth))
                aroi.setwidth(int(roiwidth))

        pass

    def appendroi(self, roi):
        self.roilist.append(roi)

    def getroi(self, index):
        return self.roilist[index]

    def showrois(self, baseimage, windowname):
        # slicepos = self.ims.slicepos
        if len(self.roilist) > 0:
            # def show(self, slicepos):
            # baseimage = self.ims.getaimage(slicepos)
            modimage = baseimage
            for r in self.roilist:
                # print("roi show")
                modimage = r.show(modimage)
            cv2.imshow(windowname, modimage)

    # the img must be binary. 0 or 1
    def measureareas(self, img):
        j = 0
        self.output = np.zeros(self.getlen())
        for r in self:
            area = r.measurearea(img)
            # print(area)
            self.output[j] = area
            # print(self.output[i,j])
            j = j + 1
        return self.output


############################################################
if __name__ == "__main__":

    # test(sys.argv[1:])
    main(sys.argv[1:])
else:
    print("__name__" + str(__name__))











