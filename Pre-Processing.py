from scipy import misc
import matplotlib.pyplot as plt
from skimage import feature
from scipy import ndimage
import cv2
import os

InputPath = 'G:/Porashuna/Note Detection Images/Fake/'
OutputPath = 'G:/Porashuna/Image Processing Training Data/Fake/'
imageNames = os.listdir(InputPath)
for i in range(len(imageNames)):
    img = misc.imread((InputPath+imageNames[i]))

    imgR=img[:,:,0]
    imgG=img[:,:,1]
    imgB=img[:,:,2]
    imgGray=.2989*imgR+.5870*imgG+.1140*imgB

    cannySigmaValue= 2
    gaussianSigmaOne=3
    gaussianSigmaTwo=1
    beta=30
    alpha=15
    imgGrayBlurred=ndimage.gaussian_filter(imgGray, gaussianSigmaOne)
    imgGrayBlurredAgain=ndimage.gaussian_filter(imgGrayBlurred, gaussianSigmaTwo)
    imgGrayEnhancedEdges=(imgGrayBlurred-imgGrayBlurredAgain)
    imgGrayWithEnhancedEdges=imgGrayBlurred+alpha*imgGrayEnhancedEdges
    imgGrayEdges = feature.canny(imgGrayWithEnhancedEdges, sigma=cannySigmaValue)
    imgCanny = imgGray + beta * imgGrayEdges

    plt.imsave(OutputPath+imageNames[i], imgCanny[111:270, 600:740])