# import the necessary packages
import argparse
import datetime
import time
import cv2

min_area = 300
contFrames=0



for x in range(1):
    #captures the video from camera
    if x!=24:
        camera = cv2.VideoCapture(0) #take from the webcam
        
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=3,nmixtures=10)
        fgbg2 = cv2.createBackgroundSubtractorMOG2(history=3,varThreshold=100,detectShadows=False)

        estrut = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        fgbg3 = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=5, decisionThreshold=0.96)


        # initialize the first frame in the video stream
        firstFrame = None
        
        temp = 0
        while True:
            # grab the current frame and initialize the occupied/unoccupied
            # text
            (grabbed, frame) = camera.read()
            if not grabbed:
                break

            fgmask = fgbg.apply(frame)

            fgmask2 = fgbg2.apply(frame)

            fgmask3 = fgbg3.apply(frame)
            fgmask3 = cv2.morphologyEx(fgmask3, cv2.MORPH_OPEN, estrut)

            text = "Unoccupied"

            # if the frame could not be grabbed, then we have reached the end
            # of the video


            # resize the frame, convert it to grayscale, and blur it
            #frame = imutils.resize(frame, width=500)
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            gray = cv2.GaussianBlur(frame, (7, 7), 0)

            # if the first frame is None, initialize it
            if firstFrame is None:
                firstFrame = gray
                continue
                # compute the absolute difference between the current frame and
                # first frame
            frameDelta = cv2.absdiff(firstFrame,gray)

            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.threshold(cv2.cvtColor(frameDelta, cv2.COLOR_BGR2GRAY), 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=3)

            diffImg = cv2.bitwise_and(frame, frame, mask=thresh)

            (image, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < min_area:
                    continue
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "Occupied"

            fgmask2 = cv2.dilate(fgmask2, None, iterations=5)
            #diffImgMOG = cv2.bitwise_and(frame, frame, mask=fgmask2)

            # draw the text and timestamp on the frame
            cv2.putText(frame, "Room Status: {}".format(text), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            # show the frame and record if the user presses a key
            cv2.imshow("Security Feed", frame)
            cv2.imshow("Gaussian Blur",gray)
            cv2.imshow("Thresh", thresh)
            cv2.imshow("Negativo", frameDelta)
            #cv2.imshow("MOG",fgmask)
            #cv2.imshow("MOG2", fgmask2)
            #cv2.imshow("Morfo", fgmask3)
            cv2.imshow("Diff Img", diffImg)
            
            temp = temp + 1
            if temp == 10:
                contFrames = contFrames+1
                #cv2.imwrite("VideoFire "+str(contFrames)+" .jpg", diffImg)
                temp = 0
            #cv2.imwrite(folder2+videoFile+"frame%d.jpg" % contFrames, fgmask2)
            #cv2.imwrite(folder3 + videoFile + "frame%d.jpg" % contFrames, fgmask3)

            key = cv2.waitKey(1) & 0xFF

            #save the frame
            #cv2.imwrite("frame%d.jpg" % count, frameDelta)



            # if the `q` key is pressed, break from the lop
            if key == ord("q"):
                break

            # cleanup the camera and close any open windows
        camera.release()
        cv2.destroyAllWindows()
