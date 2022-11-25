import warnings
warnings.filterwarnings("ignore")

print("Starting.............................................................................\n")
print("Importing Necessary Libraries\n")

import streamlit as st
import tempfile
import os
import cv2
#import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
#from pylab import rcParams
import time
from zipfile import ZipFile
#import base64
#import csv
import uuid

print("Imports Completed\n")

#st.set_page_config(layout = "wide")
st.set_page_config(page_title = "Image Utility Tools with Streamlit", page_icon = "🤖")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 340px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 340px;
        margin-left: -340px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ZipfilesPaths = {
    "imagesZipFile" : os.path.join(".", "output", "Images.zip"),
    "realTimeFeedZipFile" : os.path.join(".", "output", "LiveFeed.zip"),
    "videosZipFile" : os.path.join(".", "output", "Videos.zip"),
    "resizedImagesZipFile" : os.path.join(".", "output", "ResizedImages.zip"),
    "recordedVideosZipFile" : os.path.join(".", "output", "RecordedVideos.zip"),
    "removedBackgroundZipFile" : os.path.join(".", "output", "RemovedBackground.zip"),
    "blendedImagesDiffSizesZipFile" : os.path.join(".", "output", "BlendedImagesDiffSizes.zip"),
    "blendedImagesSameSizesZipFile" : os.path.join(".", "output", "BlendedImagesSameSizes.zip"),
    "captureImagesZipFile" : os.path.join(".", "output", "CaptureImages.zip"),
}

for path in ZipfilesPaths.values():
    if os.path.exists(path):
        os.remove(path)

outputfolderPaths = {
    "folderImages" : os.path.join(".", "output", "Images"),
    "folderRealTime" : os.path.join(".", "output", "LiveFeed"),
    "folderVideos" : os.path.join(".", "output", "Videos"),
    "folderResizedImages" : os.path.join(".", "output", "ResizedImages"),
    "folderRecordedVideos" : os.path.join(".", "output", "RecordedVideos"),
    "folderRemovedBackground" : os.path.join(".", "output", "RemovedBackground"),
    "folderBlendedImagesDiffSizes" : os.path.join(".", "output", "BlendedImagesDiffSizes"),
    "folderBlendedImagesSameSizes" : os.path.join(".", "output", "BlendedImagesSameSizes"),
    "folderCaptureImages" : os.path.join(".", "output", "CaptureImages")
}

# for path in outputfolderPaths.values():
#     for file in os.listdir(path):
#         if os.path.exists(path):
#             os.remove(os.path.join(path, file))
        
#################### SideBar ####################################################
activity = ["Take Pictures From Camera", "Blending Images of Same Sizes", "Blending Images with Different Sizes", "Remove Background of An Image", "Record Video From Camera", "Image Resize", "Write Text on Image", "Write Text on Live Feed", "Write Text on Video File"]
choice = st.sidebar.selectbox('Chose An Activity', activity)
st.sidebar.subheader("Parameters")

DEMO_VIDEO = "./samples/sampleVideo0.mp4"
# DEMO_VIDEO = "./samples/sampleVideo1.mp4"
# DEMO_VIDEO = "./samples/sampleVideo2.mp4"
DEMO_PIC = "./samples/Cars3.png"
DEMO_PIC2 = "./samples/demopic.png"

print("Processing.............\n")

################### DEFINE FUNCTIONS #########################################

@st.cache()
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
        

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
        

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

   
def main():   
    
    ##############################################################################
    ################## Streamlit #################################################
    ##############################################################################
    
    if choice == "Take Pictures From Camera":
        #################### Title ###############################################
        st.markdown("<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Take Pictures From Camera</h3>", unsafe_allow_html=True)
        st.markdown('#') # inserts empty space
        
        ###################### Select Camera #################################
        
        selectedCam = st.sidebar.selectbox("Select Camera", ("Use WebCam", "Use Other Camera"), index = 0)
        
        if selectedCam == "Use Other Camera":
            selectedCam = int(1)
        else:
            selectedCam = int(0)
        
        st.sidebar.write("Select Camera is ", selectedCam)
        
        ###################### Capture Frame From Camera ######################
        
        stframe = st.empty()
        cap = cv2.VideoCapture(selectedCam)
        #cap = cv2.VideoCapture(selectedCam, cv2.CAP_V4L2)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Width: ", width, "\n")
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Height: ", height, "\n")
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))
        print("FPS Input: ",fps_input, "\n")
        
        ###################### Take Pictures #################################
        timer = st.sidebar.number_input("Timer (seconds)")
        startStopTakingPics = st.sidebar.checkbox("Start/Stop")
        
        if startStopTakingPics:
            
            st.info("Frames Capturing in Progress....")
            
            prevTime = 0
            newTime = 0
            fps = 0
            
            try:
                while cap.isOpened():
                    ret, img = cap.read()
                    if ret:
                        #Calculations for getting FPS manually
                        newTime = time.time()
                        fps = int(1/(newTime - prevTime))
                        prevTime = newTime
                        #print("FPS: ", fps)
                        #image_np = np.array(frame)
                        
                        #stframe.image(cv2.resize(img, (width, height)), channels = 'BGR', use_column_width = True)
                        
                        # time.sleep(1/fps_input)
                        # print("Sleeping: ", 1/fps_input, "\n")
                        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        captureFramesfilename = os.path.join(outputfolderPaths["folderCaptureImages"], '{}.jpg'.format(uuid.uuid1()))
                        cv2.imwrite(captureFramesfilename, img)
                        stframe.image(cv2.resize(img, (width, height)), channels = 'BGR', use_column_width = True)
                        time.sleep(timer)
                        print(f"Waiting for {timer} second(s)")
                    else:
                        break
            except Exception as e:
                st.write("Error in Frames Capturing")
                pass
        
        cap.release()
        
        ################### Zip and Downloads ##################################     
        st.sidebar.markdown('## Output')  
        try:
            with st.spinner("Please Wait....."):
                # Zip and download
                zipObj = ZipFile(ZipfilesPaths["captureImagesZipFile"], 'w')
                if zipObj is not None:
                    # Add multiple files to the zip
                    files = os.listdir(outputfolderPaths["folderCaptureImages"])
                    for filename in files:
                        eachFile = os.path.join(outputfolderPaths["folderCaptureImages"], filename)
                        zipObj.write(eachFile)
                    zipObj.close()
        except Exception as e:
            st.write("Error in Zip and Download")
            st.error(e)
    
        with open(ZipfilesPaths["captureImagesZipFile"], 'rb') as f:
            st.sidebar.download_button('Download (.zip)', f, file_name = 'CaptureImages.zip')
    
    if choice == "Blending Images of Same Sizes":
        #################### Title ###############################################
        st.markdown("<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Blending Images of Same Sizes</h3>", unsafe_allow_html=True)
        st.markdown('#') # inserts empty space
        
        ###################### Image File Upload ##################################
        st.set_option('deprecation.showfileUploaderEncoding', False)
        st.sidebar.markdown("*Upload Two Images*")
        uploaded_files = st.sidebar.file_uploader("Choose an image file", type = ["jpg", "png"], accept_multiple_files = True)
        
        #images = images[:2]
        images = []
        
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype = np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            images.append(image)
            #st.sidebar.write("filename:", uploaded_file.name)
            #st.sidebar.image(image, channels = "BGR")
        try:
            images = images[:2]
            image1, image2 = images[0], images[1]
            #st.write("Large", imageL)
            #st.write("Small",imageS)
            
            colimage1, colimage2 = st.columns(2)
        
            with colimage1:
                st.text("Uploaded Image 1")
                resizedImage1 = cv2.resize(image1,(224, 224))
                st.image(resizedImage1, channels = "RGB")
                st.markdown("**Image 1 Dimensions**")
                h1, w1, c1 = image1.shape
                st.write(f"Height: {h1}, Width: {w1}, Channels: {c1}")
                
            with colimage2:
                st.text("Uploaded Image 2")
                resizedImage2 = cv2.resize(image2,(224, 224))
                st.image(resizedImage2, channels = "RGB")
                st.markdown("**Image 2 Dimensions**")
                h2, w2, c2 = image2.shape
                st.write(f"Height: {h2}, Width: {w2}, Channels: {c2}")
            
            # Resize Image 2 with Image 1
            image2 = cv2.resize(image2, (w1, h1))
            # st.write(image1.shape)
            # st.write(image2.shape)
                
            if st.button("Blend Images"):
                
                ### Blending the Image

                ### We will blend the values together with the formula:

                ### (image1 * alpha)  + (image2 * beta)  + gamma
                
                blended = cv2.addWeighted(src1 = image1, alpha = 0.7, src2 = image2, beta = 0.3, gamma = 0)
                blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
                filePath = os.path.join(outputfolderPaths["folderBlendedImagesSameSizes"], '{}.jpg'.format(uuid.uuid1()))
                cv2.imwrite(filePath, blended)
                
                st.image(blended, channels = "BGR")
                
            ################### Zip and Downloads ##################################     
            st.sidebar.markdown('## Output')  
            try:
                with st.spinner("Please Wait....."):
                    # Zip and download
                    zipObj = ZipFile(ZipfilesPaths["blendedImagesSameSizesZipFile"], 'w')
                    if zipObj is not None:
                        # Add multiple files to the zip
                        files = os.listdir(outputfolderPaths["folderBlendedImagesSameSizes"])
                        for filename in files:
                            eachFile = os.path.join(outputfolderPaths["folderBlendedImagesSameSizes"], filename)
                            zipObj.write(eachFile)
                        zipObj.close()
            except Exception as e:
                st.write("Error in Zip and Download")
                st.error(e)
        
            with open(ZipfilesPaths["blendedImagesSameSizesZipFile"], 'rb') as f:
                st.sidebar.download_button('Download (.zip)', f, file_name = 'BlendedImagesSameSizes.zip')
        
        except Exception as e:
            print(e)
            pass
            
    
    if choice == "Blending Images with Different Sizes":
        #################### Title ###############################################
        st.markdown("<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Blending Images with Different Sizes</h3>", unsafe_allow_html=True)
        st.markdown('#') # inserts empty space
        
        ###################### Image File Upload ##################################
        st.set_option('deprecation.showfileUploaderEncoding', False)
        st.sidebar.markdown("*Upload Two Images*")
        st.sidebar.markdown("*Always Upload Large/Main Image First*")
        uploaded_files = st.sidebar.file_uploader("Choose an image file", type = ["jpg", "png"], accept_multiple_files = True)
        
        #images = images[:2]
        images = []
        
        for uploaded_file in uploaded_files:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype = np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            images.append(image)
            #st.sidebar.write("filename:", uploaded_file.name)
            #st.sidebar.image(image, channels = "BGR")
        try:
            images = images[:2]
            imageL, imageS = images[0], images[1]
            #st.write("Large", imageL)
            #st.write("Small",imageS)
            
            colLargeImage, colSmallImage = st.columns(2)
        
            with colLargeImage:
                st.text("Uploaded Large/Main Image")
                resizedLargeImage = cv2.resize(imageL,(224, 224))
                st.image(resizedLargeImage, channels = "RGB")
                st.markdown("**Large Image Dimensions**")
                hL, wL, cL = imageL.shape
                st.write(f"Height: {hL}, Width: {wL}, Channels: {cL}")
                
            with colSmallImage:
                st.text("Uploaded Small Image")
                resizedSmallImage = cv2.resize(imageS,(224, 224))
                st.image(resizedSmallImage, channels = "RGB")
                st.markdown("**Small Image Dimensions**")
                hS, wS, cS = imageS.shape
                st.write(f"Height: {hS}, Width: {wS}, Channels: {cS}")
            
            # x = int(st.sidebar.slider('x_offset', min_value = 0, max_value = wS, value = 0))
            # y = int(st.sidebar.slider('y_offset', min_value = 0, max_value = hS, value = 0))

            # x = 300
            # y = 300
            
            if st.button("Blend Images"):
                x_offset = wL - wS
                # x_offset = wL - x
                y_offset = hL - hS
                # y_offset = hL - y
                
                ROI = imageL[y_offset:hL, x_offset:wL]
                #st.image(ROI, channels = "RGB")
                print("ROI Shape: ", ROI.shape)
                print("Small Image Shape: ", imageS.shape)
                
                imageSGray = cv2.cvtColor(imageS, cv2.COLOR_RGB2GRAY)
                #print("Small Gray Image Shape: ", imageSGray.shape)
                maskInv = cv2.bitwise_not(imageSGray)
                #print("Mask Inverse Shape: ", maskInv.shape)
                whiteBackground = np.full(imageS.shape, 255, dtype = np.uint8)
                
                backGround = cv2.bitwise_or(src1 = whiteBackground, src2 = whiteBackground, mask = maskInv)
                foreGround = cv2.bitwise_or(src1 = imageS, src2 = imageS, mask = maskInv)
                #print("Foreground Shape:", foreGround.shape)
                st.write("Foreground")
                #st.image(foreGround, channels = "RGB")
                finalROI = cv2.bitwise_or(ROI, foreGround)
                #print("Final ROI: ", finalROI.shape)
                
                largeImage = imageL
                smallImage = finalROI
                
                largeImage[y_offset : y_offset + smallImage.shape[0], x_offset : x_offset + smallImage.shape[1]] = smallImage
                print("largeImage", largeImage.shape)
                largeImage = cv2.cvtColor(largeImage, cv2.COLOR_BGR2RGB)
                filePath = os.path.join(outputfolderPaths["folderBlendedImagesDiffSizes"], '{}.jpg'.format(uuid.uuid1()))
                cv2.imwrite(filePath, largeImage)
                
                st.image(largeImage, channels = "BGR")
                
            ################### Zip and Downloads ##################################     
            st.sidebar.markdown('## Output')  
            try:
                with st.spinner("Please Wait....."):
                    # Zip and download
                    zipObj = ZipFile(ZipfilesPaths["blendedImagesDiffSizesZipFile"], 'w')
                    if zipObj is not None:
                        # Add multiple files to the zip
                        files = os.listdir(outputfolderPaths["folderBlendedImagesDiffSizes"])
                        for filename in files:
                            eachFile = os.path.join(outputfolderPaths["folderBlendedImagesDiffSizes"], filename)
                            zipObj.write(eachFile)
                        zipObj.close()
            except Exception as e:
                st.write("Error in Zip and Download")
                st.error(e)
        
            with open(ZipfilesPaths["blendedImagesDiffSizesZipFile"], 'rb') as f:
                st.sidebar.download_button('Download (.zip)', f, file_name = 'BlendedImagesDiffSizesZipFile.zip')
        except Exception as e:
            print(e)
            pass

        
    if choice == "Remove Background of An Image":
        segmentor = SelfiSegmentation()
        #################### Title ###############################################
        st.markdown("<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Remove Background of An Image</h3>", unsafe_allow_html=True)
        st.markdown('#') # inserts empty space
        
       ###################### Image File Upload ##################################
        st.set_option('deprecation.showfileUploaderEncoding', False)
        uploaded_file = st.file_uploader("Choose an image file", type = ["jpg", "png"])

        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype = np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            #resized = cv2.resize(img,(224, 224))
            # Now do something with the image! For example, let's display it:
            st.sidebar.text("Uploaded Pic")
            st.sidebar.image(image, channels = "BGR")
        else:
            image = cv2.imread(DEMO_PIC2)
            st.sidebar.text("DEMO Pic")
            st.sidebar.image(image, channels = "BGR")
            #st.info("Upload an Image")
            
        st.sidebar.markdown(""" **Select Red, Green and Blue Values for Background Color** """)
        st.sidebar.markdown(""" **Default is 'White'** """)
        
        redColorValue = st.sidebar.slider('Red', min_value = 0, max_value = 255, value = 255) 
        greenColorValue = st.sidebar.slider('Green', min_value = 0, max_value = 255, value = 255)
        blueColorValue = st.sidebar.slider('Blue', min_value = 0, max_value = 255, value = 255)
            
        if st.button("Remove Background"):
            try:
                #img = cv2.resize(image, (640, 480))   
                # green = (0, 255, 0)
                backgroundColor = (redColorValue, greenColorValue, blueColorValue)
                imgNoBg = segmentor.removeBG(image, backgroundColor, threshold = 0.50)
                # show Image
                st.image(imgNoBg, channels = "BGR")
                filePath = os.path.join(outputfolderPaths["folderRemovedBackground"], '{}.jpg'.format(uuid.uuid1()))
                cv2.imwrite(filePath, imgNoBg)
            except Exception as e:
                st.write("Error in Remove Background")
                print(e)
                pass
            
        ################### Zip and Downloads ##################################     
        st.sidebar.markdown('## Output')  
        try:
            with st.spinner("Please Wait....."):
                # Zip and download
                zipObj = ZipFile(ZipfilesPaths["removedBackgroundZipFile"], 'w')
                if zipObj is not None:
                    # Add multiple files to the zip
                    files = os.listdir(outputfolderPaths["folderRemovedBackground"])
                    for filename in files:
                        eachFile = os.path.join(outputfolderPaths["folderRemovedBackground"], filename)
                        zipObj.write(eachFile)
                    zipObj.close()
        except Exception as e:
            st.write("Error in Zip and Download")
            st.error(e)
        
        with open(ZipfilesPaths["removedBackgroundZipFile"], 'rb') as f:
            st.sidebar.download_button('Download (.zip)', f, file_name = 'RemovedBackground.zip')

    
    if choice == "Record Video From Camera":
        #################### Title ###########################################
        st.markdown("<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Record Video From WebCam/Other Camera</h3>", unsafe_allow_html=True)
        st.markdown('#') # inserts empty space
        
        ###################### Select Camera #################################
        
        selectedCam = st.sidebar.selectbox("Select Camera", ("Use WebCam", "Use Other Camera"), index = 0)
        
        if selectedCam == "Use Other Camera":
            selectedCam = int(1)
        else:
            selectedCam = int(0)
        
        #st.sidebar.write("Select Camera is ", selectedCam)
        
        ###################### Capture Frame From Camera ######################
        
        stframe = st.empty()
        cap = cv2.VideoCapture(selectedCam)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Width: ", width, "\n")
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Height: ", height, "\n")
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))
        print("FPS Input: ",fps_input, "\n")
        
        ###################### Record Video #################################
        startStopCam = st.sidebar.checkbox("Record/Stop")
        if startStopCam:
            st.info("Recording in Progress....")
            
            prevTime = 0
            newTime = 0
            fps = 0
            
            #filename = './output/WartermarkLiveFeed/{}.mp4'.format(uuid.uuid1())
            recordedVideofilename = os.path.join(outputfolderPaths["folderRecordedVideos"], 'recordedVideo.mp4')
            #realTimeVideofilename = './output/WartermarkLiveFeed/LiveFeed.mp4'
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            resolution = (width, height)
            
            VideoOutPut = cv2.VideoWriter(recordedVideofilename, codec, fps_input, resolution)
            
            try:
                while cap.isOpened():
                    ret, img = cap.read()
                    if ret:
                        #Calculations for getting FPS manually
                        newTime = time.time()
                        fps = int(1/(newTime - prevTime))
                        prevTime = newTime
                        #print("FPS: ", fps)
                        #image_np = np.array(frame)
                        
                        #stframe.image(cv2.resize(img, (width, height)), channels = 'BGR', use_column_width = True)
                        
                        time.sleep(1/fps_input)
                        print("Sleeping: ", 1/fps_input, "\n")
                        VideoOutPut.write(img)
                        stframe.image(cv2.resize(img, (width, height)), channels = 'BGR', use_column_width = True)
                    else:
                        break
            except Exception as e:
                st.write("Error in Recording")
                pass
                
            cap.release()
            VideoOutPut.release()
        
        ################### Zip and Downloads ##################################     
        st.sidebar.markdown('## Output')  
        try:
            with st.spinner("Please Wait....."):
                # Zip and download
                zipObj = ZipFile(ZipfilesPaths["recordedVideosZipFile"], 'w')
                if zipObj is not None:
                    # Add multiple files to the zip
                    files = os.listdir(outputfolderPaths["folderRecordedVideos"])
                    for filename in files:
                        eachFile = os.path.join(outputfolderPaths["folderRecordedVideos"], filename)
                        zipObj.write(eachFile)
                    zipObj.close()
        except Exception as e:
            st.write("Error in Zip and Download")
            st.error(e)
        
        with open(ZipfilesPaths["recordedVideosZipFile"], 'rb') as f:
            st.sidebar.download_button('Download (.zip)', f, file_name = 'RecordedVideos.zip')
        
        #Print File Names
        try:
            st.sidebar.write(recordedVideofilename)
            print("Real Time .mp4 File Name: ", recordedVideofilename, "\n")
        except:
            pass
        
    if choice == "Image Resize":
        #################### Title ###########################################
        st.markdown("<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Resize Image</h3>", unsafe_allow_html=True)
        st.markdown('#') # inserts empty space
        
        ###################### Image File Upload #################################
        st.set_option('deprecation.showfileUploaderEncoding', False)
        uploaded_file = st.file_uploader("Choose an image file", type = ["jpg", "png"])

        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype = np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            #resized = cv2.resize(img,(224, 224))
            # Now do something with the image! For example, let's display it:
            st.sidebar.text("Uploaded Pic")
            st.sidebar.image(image, channels = "BGR")
        else:
            image = cv2.imread(DEMO_PIC)
            st.sidebar.text("DEMO Pic")
            st.sidebar.image(image, channels = "BGR")
            #st.info("Upload an Image")
        
        img_height, img_width, channels = image.shape
        st.write("Given Image Dimensions are: ", f"Height: {img_height}, Width: {img_width}, Channels: {channels}")
        
        #print(img.shape)
        
        #################### Resize Imag ########################################
        st.sidebar.subheader("Parameters To Set")
        widthToResize = None
        heightToResize = None
        input = st.sidebar.selectbox("Select Width or Height", ("Width", "Height"))
        
        if input == "Width":
            widthToResize = st.sidebar.number_input("Enter Width", None)
            img_width = int(widthToResize)
            img_height = None
            st.write("Widht To Resize: ", img_width)
           
        
        if input == "Height":
            heightToResize = st.sidebar.number_input("Enter Height", None)
            img_height = int(heightToResize)
            img_width = None
            st.write("Widht To Height: ", img_height)
            
        
        if st.button("Resize"):
            try:
                resized = image_resize(image, width = img_width, height = img_height, inter = cv2.INTER_AREA)
                st.markdown("**Resized Image**")
                st.image(resized, channels = "BGR")
                filePath = filePath = os.path.join(outputfolderPaths["folderResizedImages"], '{}.jpg'.format(uuid.uuid1()))
                cv2.imwrite(filePath, resized)
                resizedHeight, resizedWidth = resized.shape[:2]
                st.write("resized Height: ", resizedHeight, "resized Width: ", resizedWidth)
                st.success('Done')
            except Exception as e:
                st.write("Error In Resizing, Please check entered parameters")
                print(e)
                pass
                
        ################### Zip and Downloads ##################################      
        st.sidebar.markdown('## Output')   
        try:
            with st.spinner("Please Wait....."):
                # Zip and download
                zipObj = ZipFile(ZipfilesPaths["resizedImagesZipFile"], 'w')
                if zipObj is not None:
                    # Add multiple files to the zip
                    files = os.listdir(outputfolderPaths["folderResizedImages"])
                    #print("folder Images:", folderImages)
                    for filename in files:
                        eachFile = os.path.join(outputfolderPaths["folderResizedImages"], filename)
                        zipObj.write(eachFile)
                    zipObj.close()
        except Exception as e:
            st.write("Error in Zip and Download")
            st.error(e)
        
        try:
            with open(ZipfilesPaths["resizedImagesZipFile"], 'rb') as f:
                st.sidebar.download_button('Download (.zip)', f, file_name = 'ResizedImages.zip')
        except Exception as e:
            print(e)

    
    if choice == "Write Text on Image":
        
        #################### Title ###############################################
        st.markdown("<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Write Text on Image</h3>", unsafe_allow_html=True)
        st.markdown('#') # inserts empty space
        
        ###################### Image File Upload #################################
        st.set_option('deprecation.showfileUploaderEncoding', False)
        uploaded_file = st.file_uploader("Choose an image file", type = ["jpg", "png"])

        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype = np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
            #resized = cv2.resize(img,(224, 224))
            # Now do something with the image! For example, let's display it:
            st.sidebar.text("Uploaded Pic")
            st.sidebar.image(img, channels = "BGR")
        else:
            img = cv2.imread(DEMO_PIC)
            st.sidebar.text("DEMO Pic")
            st.sidebar.image(img, channels = "BGR")
            #st.info("Upload an Image")
        
        height, width, channels = img.shape
        
        #print(img.shape)
            
        #################### Parameters to setup ########################################
        st.sidebar.subheader("Parameters To Set")
        
        text = st.sidebar.text_input('Enter Text To Write On Image: ', 'Text to Write')
        st.sidebar.markdown(""" **x, y are the points where you would like to put text** """)
        x = st.sidebar.slider('x', min_value = 0, max_value = width, value = 0)
        y = st.sidebar.slider('y', min_value = 0, max_value = height, value = 0)

        st.sidebar.markdown(""" **Select Font** """)
        fontList = {0 : cv2.FONT_HERSHEY_SIMPLEX, 1 : cv2.FONT_HERSHEY_PLAIN, 2 : cv2.FONT_HERSHEY_DUPLEX, 3 : cv2.FONT_HERSHEY_COMPLEX, 4 : cv2.FONT_HERSHEY_TRIPLEX, 5 : cv2.FONT_HERSHEY_COMPLEX_SMALL, 6 : cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 7 : cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 8 : cv2.FONT_ITALIC}
        option = st.sidebar.selectbox('Select Font', ([value for key, value in fontList.items()]))
        st.sidebar.write('You selected:', option)

        st.sidebar.markdown(""" **Select Font Size** """)
        fontSize = st.sidebar.slider('Font Size', min_value = 0, max_value = 5, value = 1)

        st.sidebar.markdown(""" **Select Red, Green and Blue Values for Color** """)
        redColorValue = st.sidebar.slider('Red', min_value = 0, max_value = 255, value = 15) 
        greenColorValue = st.sidebar.slider('Green', min_value = 0, max_value = 255, value = 85)
        blueColorValue = st.sidebar.slider('Blue', min_value = 0, max_value = 255, value = 245)

        st.sidebar.markdown(""" **Select Thickness** """)
        thicknessValue = st.sidebar.slider('Thickness', min_value = 0, max_value = 10, value = 1)

        #################### /Parameters to setup #######################################

        ###################### Write To Image #################################
        
        #Coordinates for Rectangle
        x = width//2
        y = height//2

        # Width and height
        w = width//4
        h = height//4
        
        if st.button("Write Text on Image"):
            with st.spinner(text = 'In progress..............'):
                try:
                    # Draw Rectangle
                    # cv2.rectangle(img, (x, y), (x + w, y + h), color = (redColorValue, greenColorValue, blueColorValue), thickness = thicknessValue)

                    # #Draw Circle
                    # cv2.circle(img, (200, 200), 80, (0, 255, 0), -1)

                    #Put Text
                    #cv2.putText(img, text, (x, y), FONT, size, color, thickness)
                    cv2.putText(img, text, (x, y), option, fontSize, (redColorValue, greenColorValue, blueColorValue), thicknessValue)
                    st.image(img, channels = "BGR")
                    filePath = os.path.join(outputfolderPaths["folderImages"], '{}.jpg'.format(uuid.uuid1())) 
                    cv2.imwrite(filePath, img)
                    st.success('Done')
                except Exception as e:
                    st.write("Error in Writing to Images")
                    st.error(e)
                    pass
                
        ################### Zip and Downloads ##################################      
        st.sidebar.markdown('## Output')   
        try:
            with st.spinner("Please Wait....."):
                # Zip and download
                zipObj = ZipFile(ZipfilesPaths['imagesZipFile'], 'w')
                if zipObj is not None:
                    # Add multiple files to the zip
                    files = os.listdir(outputfolderPaths["folderImages"])
                    #print("folder Images:", folderImages)
                    for filename in files:
                        eachFile = os.path.join(outputfolderPaths["folderImages"], filename)
                        zipObj.write(eachFile)
                    zipObj.close()
        except Exception as e:
            st.write("Error in Zip and Download")
            st.error(e)
        
        try:
            with open(ZipfilesPaths['imagesZipFile'], 'rb') as f:
                st.sidebar.download_button('Download (.zip)', f, file_name = 'Images.zip')
        except Exception as e:
            print(e)
                
                
    if choice == "Write Text on Live Feed":
        
        #################### Title ###########################################
        st.markdown("<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Write Text on Live Feed</h3>", unsafe_allow_html=True)
        st.markdown('#') # inserts empty space
        
        ###################### Select Camera #################################
        
        selectedCam = st.sidebar.selectbox("Select Camera", ("Use WebCam", "Use Other Camera"), index = 0)
        
        if selectedCam == "User Other Camera":
            selectedCam = int(1)
        else:
            selectedCam = int(0)
        
        #st.sidebar.write("Select Camera is ", selectedCam)
        
        ###################### Capture Frame From Camera ######################
        
        stframe = st.empty()
        cap = cv2.VideoCapture(selectedCam)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("Width: ", width, "\n")
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Height: ", height, "\n")
        fps_input = int(cap.get(cv2.CAP_PROP_FPS))
        print("FPS Input: ",fps_input, "\n")
        
        #################### Parameters to setup ########################################
        st.sidebar.subheader("Parameters To Set")
        
        text = st.sidebar.text_input('Enter Text To Write: ', 'Write Text')
        st.sidebar.markdown(""" **x, y are the points where you would like to put text** """)
        x = st.sidebar.slider('x', min_value = 0, max_value = width, value = 0)
        y = st.sidebar.slider('y', min_value = 0, max_value = height, value = 0)

        st.sidebar.markdown(""" **Select Font** """)
        fontList = {0 : cv2.FONT_HERSHEY_SIMPLEX, 1 : cv2.FONT_HERSHEY_PLAIN, 2 : cv2.FONT_HERSHEY_DUPLEX, 3 : cv2.FONT_HERSHEY_COMPLEX, 4 : cv2.FONT_HERSHEY_TRIPLEX, 5 : cv2.FONT_HERSHEY_COMPLEX_SMALL, 6 : cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 7 : cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 8 : cv2.FONT_ITALIC}
        option = st.sidebar.selectbox('Select Font', ([value for key, value in fontList.items()]))
        st.sidebar.write('You selected:', option)

        st.sidebar.markdown(""" **Select Font Size** """)
        fontSize = st.sidebar.slider('Font Size', min_value = 0, max_value = 5, value = 1)

        st.sidebar.markdown(""" **Select Red, Green and Blue Values for Color** """)
        redColorValue = st.sidebar.slider('Red', min_value = 0, max_value = 255, value = 15) 
        greenColorValue = st.sidebar.slider('Green', min_value = 0, max_value = 255, value = 85)
        blueColorValue = st.sidebar.slider('Blue', min_value = 0, max_value = 255, value = 245)

        st.sidebar.markdown(""" **Select Thickness** """)
        thicknessValue = st.sidebar.slider('Thickness', min_value = 0, max_value = 10, value = 1)

        #################### /Parameters to setup #######################################
        
        ###################### Write To Video #################################
        startStopCam = st.checkbox("Record/Stop")
        if startStopCam:
            st.info("Recording in Progress....")
            
            prevTime = 0
            newTime = 0
            fps = 0
            
            #filename = './output/WartermarkLiveFeed/{}.mp4'.format(uuid.uuid1())
            realTimeVideofilename = os.path.join(outputfolderPaths["folderRealTime"], 'LiveFeed.mp4')
            #realTimeVideofilename = './output/WartermarkLiveFeed/LiveFeed.mp4'
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            resolution = (width, height)
            
            VideoOutPut = cv2.VideoWriter(realTimeVideofilename, codec, fps_input, resolution)
            
            while cap.isOpened():
                ret, img = cap.read()
                if ret:
                    #Calculations for getting FPS manually
                    newTime = time.time()
                    fps = int(1/(newTime - prevTime))
                    prevTime = newTime
                    #print("FPS: ", fps)
                    #image_np = np.array(frame)
                    
                    #stframe.image(cv2.resize(img, (width, height)), channels = 'BGR', use_column_width = True)
                    
                    try: 
                        # Draw Rectangle
                        # cv2.rectangle(img, (x, y), (x + w, y + h), color = (redColorValue, greenColorValue, blueColorValue), thickness = thicknessValue)

                        # #Draw Circle
                        # cv2.circle(img, (200, 200), 80, (0, 255, 0), -1)

                        #Put Text
                        #cv2.putText(img, text, (x, y), FONT, size, color, thickness)
                        cv2.putText(img, text, (x, y), option, fontSize, (redColorValue, greenColorValue, blueColorValue), thicknessValue)
                        
                        #st.success('Done')
                    except Exception as e:
                        st.write("Error in Writing to Real Time Feed")
                        st.error(e)
                        pass
                    
                    time.sleep(1/fps_input)
                    print("Sleeping: ", 1/fps_input, "\n")
                    VideoOutPut.write(img)
                    stframe.image(cv2.resize(img, (width, height)), channels = 'BGR', use_column_width = True)
                else:
                    break
            
            cap.release()
            VideoOutPut.release()
        
        ################### Zip and Downloads ##################################     
        st.sidebar.markdown('## Output')  
        try:
            with st.spinner("Please Wait....."):
                # Zip and download
                zipObj = ZipFile(ZipfilesPaths["realTimeFeedZipFile"], 'w')
                if zipObj is not None:
                    # Add multiple files to the zip
                    files = os.listdir(outputfolderPaths["folderRealTime"])
                    for filename in files:
                        eachFile = os.path.join(outputfolderPaths["folderRealTime"], filename)
                        zipObj.write(eachFile)
                    zipObj.close()
        except Exception as e:
            st.write("Error in Zip and Download")
            st.error(e)
        
        with open(ZipfilesPaths["realTimeFeedZipFile"], 'rb') as f:
            st.sidebar.download_button('Download (.zip)', f, file_name = 'LiveFeed.zip')
        
        #Print File Names
        try:
            st.sidebar.write(realTimeVideofilename)
            print("Real Time .mp4 File Name: ", realTimeVideofilename, "\n")
        except:
            pass
    
       
    if choice == "Write Text on Video File":
        
        #################### Title ###########################################
        st.markdown("<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Write Text on Video File</h3>", unsafe_allow_html=True)
        st.markdown('#') # inserts empty space
        
        ###################### Image File Upload #############################
        st.set_option('deprecation.showfileUploaderEncoding', False)
        stframe = st.empty()
        #video_file_buffer = st.sidebar.file_uploader("Upload a video", type = [ "mp4", "mov",'avi','asf', 'm4v' ])
        video_file_buffer = st.sidebar.file_uploader("Upload a video", type = [ "mp4", "m4v"])
        tffile = tempfile.NamedTemporaryFile(delete = False)

        if not video_file_buffer:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tffile.name = DEMO_VIDEO
            st.sidebar.markdown("**DEMO Video**")
            st.sidebar.video(tffile.name)
        else:
            tffile.write(video_file_buffer.read())
            vid = cv2.VideoCapture(tffile.name)
            st.sidebar.markdown("**Uploaded Video**")
            st.sidebar.video(tffile.name)
            
        ###################### Capture Frame From Video File ######################
            
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        #st.write(width)
        print("Width: ", width, "\n")
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #st.write(height)
        print("Height: ", height, "\n")
        fps_input = int(vid.get(cv2.CAP_PROP_FPS))
        #st.write(fps_input)
        print("FPS Input: ",fps_input, "\n")
        
        #st.sidebar.markdown("## Output")
        #st.sidebar.text("Default/Uploaded Video")
        # st.sidebar.markdown("**Default/Uploaded Video**")
        # st.sidebar.video(tffile.name)
        
        #################### Parameters to setup ########################################
        st.sidebar.subheader("Parameters To Set")
        
        text = st.sidebar.text_input('Enter Text To Write On Video File: ', 'Write Text')
        st.sidebar.markdown(""" **x, y are the points where you would like to put text** """)
        x = st.sidebar.slider('x', min_value = 0, max_value = width, value = 0)
        y = st.sidebar.slider('y', min_value = 0, max_value = height, value = 0)

        st.sidebar.markdown(""" **Select Font** """)
        fontList = {0 : cv2.FONT_HERSHEY_SIMPLEX, 1 : cv2.FONT_HERSHEY_PLAIN, 2 : cv2.FONT_HERSHEY_DUPLEX, 3 : cv2.FONT_HERSHEY_COMPLEX, 4 : cv2.FONT_HERSHEY_TRIPLEX, 5 : cv2.FONT_HERSHEY_COMPLEX_SMALL, 6 : cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 7 : cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 8 : cv2.FONT_ITALIC}
        option = st.sidebar.selectbox('Select Font', ([value for key, value in fontList.items()]))
        st.sidebar.write('You selected:', option)

        st.sidebar.markdown(""" **Select Font Size** """)
        fontSize = st.sidebar.slider('Font Size', min_value = 0, max_value = 5, value = 1)

        st.sidebar.markdown(""" **Select Red, Green and Blue Values for Color** """)
        redColorValue = st.sidebar.slider('Red', min_value = 0, max_value = 255, value = 15) 
        greenColorValue = st.sidebar.slider('Green', min_value = 0, max_value = 255, value = 85)
        blueColorValue = st.sidebar.slider('Blue', min_value = 0, max_value = 255, value = 245)

        st.sidebar.markdown(""" **Select Thickness** """)
        thicknessValue = st.sidebar.slider('Thickness', min_value = 0, max_value = 10, value = 1)

        #################### /Parameters to setup #######################################
        
        ###################### Dashbord Stuff ####################################
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.markdown("**Frame Rate**")
            kpi1_text = st.markdown("0")

        with kpi2:
            st.markdown("**Text To Write**")
            kpi2_text = st.markdown("0")
            kpi2 = "Change Later"

        with kpi3:
            st.markdown("**Image Width**")
            kpi3_text = st.markdown("0")
        
        with kpi4:
            st.markdown("**Image Height**")
            kpi4_text = st.markdown("0")

        st.markdown("<hr/>", unsafe_allow_html=True)
        
        ###################### Write To Video File ##############################
        startStop = st.checkbox("Start/Stop")
        if startStop:
            #st.info("In Progress....")
            
            prevTime = 0
            newTime = 0
            fps = 0
            
            #filename = './output/WartermarkVideos/{}.mp4'.format(uuid.uuid1())
            videofilename = os.path.join(outputfolderPaths["folderVideos"], 'outVideoFile.mp4')
            #videofilename = './output/Detection_From_Videos/outVideoFile.mp4'
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            resolution = (width, height)
           
            VideoOutPut = cv2.VideoWriter(videofilename, codec, fps_input, resolution)
            
            while vid.isOpened():
                ret, img = vid.read()
                if ret:
                    #Calculations for getting FPS manually
                    newTime = time.time()
                    fps = int(1/(newTime - prevTime))
                    prevTime = newTime
                    
                    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #img_height, img_width, _ = img.shape
                    #print("Image Shape: \n", img.shape)
                    # print("Type: \n", type(img))
                    
                    try: 
                       # Draw Rectangle
                        # cv2.rectangle(img, (x, y), (x + w, y + h), color = (redColorValue, greenColorValue, blueColorValue), thickness = thicknessValue)

                        # #Draw Circle
                        # cv2.circle(img, (200, 200), 80, (0, 255, 0), -1)

                        #Put Text
                        #cv2.putText(img, text, (x, y), FONT, size, color, thickness)
                        cv2.putText(img, text, (x, y), option, fontSize, (redColorValue, greenColorValue, blueColorValue), thicknessValue)
                        
                        #st.success('Done')
                    except Exception as e:
                        st.write("Error in Writing to Video File")
                        st.error(e)
                        pass
                        
                    time.sleep(1/fps_input)
                    print("Sleeping: ", 1/fps_input, "\n")
                    VideoOutPut.write(img)        
                            
                    #Dashboard            
                    kpi1_text.write(f"<h3 style='text-align: left; color: red;'>{int(fps_input)}</h3>", unsafe_allow_html=True)
                    #kpi1_text.write(f"<h4 style='text-align: left; color: red;'>{int(fps)}</h4>", unsafe_allow_html=True)
                    kpi2_text.write(f"<h3 style='text-align: left; color: red;'>{kpi2}</h3>", unsafe_allow_html=True)
                    kpi3_text.write(f"<h3 style='text-align: left; color: red;'>{width}</h3>", unsafe_allow_html=True)
                    kpi4_text.write(f"<h3 style='text-align: left; color: red;'>{height}</h3>", unsafe_allow_html=True)
                    #Display on Dashboard
                    stframe.image(cv2.resize(img, (width, height)), channels = 'BGR', use_column_width = True)
                    
                else:
                    break
        
            vid.release()
            VideoOutPut.release()
            
        ################### Zip and Downloads ##################################
        st.sidebar.markdown('## Output') 
        try:
            with st.spinner("Please Wait....."):
                # Zip and download
                zipObj = ZipFile(ZipfilesPaths["videosZipFile"], 'w')
                if zipObj is not None:
                    # Add multiple files to the zip
                    files = os.listdir(outputfolderPaths["folderVideos"])
                    for filename in files:
                        eachFile = os.path.join(outputfolderPaths["folderVideos"], filename)
                        zipObj.write(eachFile)
                    zipObj.close()
                    
        except Exception as e:
            st.write("Error in Zip and Download")
            st.error(e)
        
        with open(ZipfilesPaths["videosZipFile"], 'rb') as f:
            st.sidebar.download_button('Download (.zip) Video2Text', f, file_name = 'Videos.zip')
                
        #Print File Names
        try:
            st.write(videofilename)
            print("Video .mp4 File Name: ", videofilename, "\n")
        except:
            pass
            
    print("END ............................................................................")
          


if __name__ == "__main__":
    main()