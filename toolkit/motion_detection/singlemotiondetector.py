import numpy as np
import imutils
import cv2

class SingleMotionDetector:
	def __init__(self, accumWeight=0.5):
		# initialize accumulated weight factor
		self.accumWeight = accumWeight

		# initialize background model
		self.bg = None

	def update(self, image):
		if self.bg is None:
			self.bg = image.copy().astype("float")
			return

		# the weighted average between bg, frame and accumweight factor
		cv2.accumulateWeighted(image, self.bg, self.accumWeight)

	def detect(self, image, treshVal=25):
		# get the abs difference between the background model and the image passed in
		# threshold delta image
		delta = cv2.absdiff(self.bg.astype("uint8"), image)
		thresh = cv2.threshold(delta, treshVal, 255, cv2.THRESH_BINARY)[1]

		# erosion & dilations to remove noise
		thresh = cv2.erode(thresh, None, iterations=2)
		thresh = cv2.dilate(thresh, None, iterations=2)

		# find contours in the thresholded image and initialize
		# min and max bounding box regions for motion
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		(minX, minY) = (np.inf, np.inf)
		(maxX, maxY) = (-np.inf, -np.inf)

		# if no contours, return None
		if len(cnts) == 0:
			return None

		for c in cnts:
			# compute the bounding box of the contour and use it to
			# update the minimum and maximum bounding box regions
			(x, y, w, h) = cv2.boundingRect(c)
			(minX, minY) = (min(minX, x), min(minY, y))
			(maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))

		# return a tuple of the thresholded image along
		# with bounding box
		return (thresh, (minX, minY, maxX, maxY))