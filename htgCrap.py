import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
class htgCrap:
    def __init__(self,read_image_path):
        self.GreysIMG = cv2.cvtColor(read_image_path,cv2.COLOR_BGR2GRAY)
        
    def binayIMG(self,L):
        def calcGrayHist(I):
            # 计算灰度直方图
            h, w = I.shape[:2]
            I = np.array(Image.fromarray(I).resize((128,int(128*(h/w)))))
            h, w = I.shape[:2]
            grayHist = np.zeros([256], np.uint64)
            theHightPoint = [0,0]#1.直方图纵坐标，2.直方图横坐标
            for i in range(h):
                for j in range(w):
                    grayHist[I[i][j]] += 1
                    if grayHist[I[i][j]] > theHightPoint[0]:
                        theHightPoint[1] = I[i][j]
                        theHightPoint[0] = grayHist[I[i][j]]
            theLowPoint = theHightPoint[1]
            gradientNumber = 0
            for i in range(1,int(theHightPoint[1]/2)):
                index = theHightPoint[1]-i*2
                newGradient = grayHist[theHightPoint[1]]-grayHist[index]
                if newGradient > gradientNumber  and grayHist[index]< 5:
                    theLowPoint = index
                    gradientNumber = newGradient
                    break
            return theLowPoint
        theLowColorPoint = calcGrayHist(L)
        GrayImage = np.array(L)
        _,thresh1=cv2.threshold(GrayImage,theLowColorPoint-int(theLowColorPoint/10),255,cv2.THRESH_BINARY)
        #中值滤波
        thresh1 = cv2.medianBlur(thresh1,5)
        return thresh1
    
    def crapFormula(self):#切割表达式
        self.BCIMG = self.binayIMG(self.GreysIMG)
        def calcHorizenFormula(I):
            h, w = I.shape[:2]
            I = cv2.bitwise_not(I)
            grayHist = np.zeros([w], np.uint64)
            grayHist = np.sum(I,axis=1)
            for i in range(h):
                grayHist[i] = grayHist[i]/255
            total = 0
            total_num = 0
            for i in range(h):
                if grayHist[i] > 0:
                    total_num+=1
                    total += grayHist[i]
            total /= total_num
            total /= 2
            x = np.arange(h)
            FormulaList = []
            startLowPoint = 0
            endLowPoint = 0
            pointHeight = 0
            traceFlag = False
            for i in range(h):
                if traceFlag == False:
                    if grayHist[i]>0:
                        traceFlag = True
                        startLowPoint = i
                else:
                    if pointHeight < grayHist[i]:
                        pointHeight = grayHist[i]
                    if i+1 >= len(grayHist)-1:
                        traceFlag = False
                        endLowPoint = i
                        continuePointNumber = 0
                        pointHeight = 0
                        FormulaList.append([startLowPoint,endLowPoint])
                        break
                    if grayHist[i]==0 and grayHist[i+1] == 0:
                        traceFlag = False
                        endLowPoint = i
                        if pointHeight > total:
                            FormulaList.append([max(startLowPoint-2,0),min(endLowPoint+2,h-1)+2])
                        continuePointNumber = 0
                        pointHeight = 0
            return FormulaList
        FL = calcHorizenFormula(self.BCIMG)
        formulaList = []
        for i in FL:
            h,w = self.BCIMG.shape
            cimg = self.BCIMG[i[0]:i[1],0:w]
            formulaList.append(cimg)
        return formulaList
    
    def crapNumberForStandard(self):#横向切割字符
        self.FL = np.array(self.crapFormula())
        def calcNumStandPoint(I):
            # 计算灰度直方图
            h, w = I.shape[:2]
            I = cv2.bitwise_not(I)
            grayHist = np.zeros([h], np.uint64)
            grayHist = np.sum(I,axis=0)
            for i in range(w):
                grayHist[i] = grayHist[i]/255
            x = np.arange(w)
            FormulaList = []
            startLowPoint = 0
            endLowPoint = 0
            pointHeight = 0
            continuePointNumber = 0
            traceFlag = False
            total = 0
            total_num=0
            for i in range(w):
                if grayHist[i] > 0:
                    total_num+=1
                    total += grayHist[i]
            total /= total_num
            for i in range(w):
                if traceFlag == False:
                    if grayHist[i]>0:
                        continuePointNumber = 0
                        traceFlag = True
                        startLowPoint = i
                else:
                    continuePointNumber+=1
                    if pointHeight < grayHist[i]:
                        pointHeight = grayHist[i]
                    if i+1 >= len(grayHist)-1:
                        traceFlag = False
                        endLowPoint = i
                        continuePointNumber = 0
                        pointHeight = 0
                        FormulaList.append([startLowPoint,endLowPoint])
                        break
                    if grayHist[i]<=0 and grayHist[i+1] <= 0:
                        traceFlag = False
                        endLowPoint = i
                        if (pointHeight + continuePointNumber) >= total:
                            FormulaList.append([startLowPoint,endLowPoint])
                        continuePointNumber = 0
                        pointHeight = 0
            return FormulaList
        FormulaList = []
        for fl in self.FL:
            numImgList = []
            h,w = fl.shape
            NumberPointList = calcNumStandPoint(fl)
            for i in NumberPointList:
                cropNumberImg = fl[0:h,i[0]:i[1]]
                numImgList.append(cropNumberImg)
            FormulaList.append(numImgList)
        return FormulaList
    
    def crapNumberForHorizen(self):#纵向切割字符
        self.NBList =  self.crapNumberForStandard()
        def calcNumForHor(I):
            # 计算灰度直方图
            h, w = I.shape[:2]
            I = cv2.bitwise_not(I)
            grayHist = np.zeros([w], np.uint64)
            grayHist = np.sum(I,axis=1)
            for i in range(h):
                grayHist[i] = grayHist[i]/255
            total = 0
            total_num = 0
            for i in range(h):
                if grayHist[i] > 0:
                    total_num+=1
                    total += grayHist[i]
            total /= total_num
            total /= 4
            #计算平均投影过滤阈值
            x = np.arange(h)
            FormulaList = []
            startLowPoint = 0
            endLowPoint = 0
            tempStartPoint = 0
            tempEndPoint = 0
            startTraceFlag = False
            endTraceFlag = False
            #第二代利用
           #第一代求纵向距离算法
            for i in range(h):
                startIndex = i
                endIndex = h-i-1
                if startTraceFlag == False:
                    if grayHist[startIndex] > total:
                        startTraceFlag = True
                        startLowPoint = startIndex
                if endTraceFlag == False:
                    if grayHist[endIndex] > total:
                        endTraceFlag = True
                        endLowPoint = endIndex
            return max(startLowPoint-int(h*0.1),0),min(endLowPoint+int(h*0.1),h-1)
        FormulaList = []
        for fl in self.NBList:
            singleNumber = []
            if fl == []:
                continue
            w = len(fl[0])
            for nb in fl:
                sp,ep = calcNumForHor(nb)
                ig = nb[sp:ep,0:w]
                singleNumber.append(ig)
            FormulaList.append(singleNumber)
        return FormulaList
    
    def crapNumber(self,input_size):#归一化字符
        FL = self.crapNumberForHorizen()
        def ChangeImgSize(L,size):
            newCanvas = Image.new('L',(size,size))
            L = cv2.bitwise_not(L)
            h,w = np.array(L).shape
            newW = int(size*0.8)
            newH = int(size*0.8)
            newX = 0
            newY = 0
            if h>w:
                newW = int(size*(w/h))
            else:
                newH = int(size*(h/w))
            newX = int((size-newW)/2)
            newY = int((size-newH)/2)
            newL2 = Image.fromarray(L).resize((newW,newH))
            newCanvas.paste(newL2,(newX,newY))
            L = np.array(newCanvas)
            L = cv2.blur(L,(5,5))#羽化
            ret, L = cv2.threshold(L, 1, 255,cv2.THRESH_BINARY)
            return L
        formulaList = []
        for fl in FL:
            singleNumber = []
            for nb in fl:
                try:
                    ig = ChangeImgSize(nb,input_size)
                except Exception:
                    pass
                singleNumber.append(ig)
            formulaList.append(singleNumber)
        return formulaList