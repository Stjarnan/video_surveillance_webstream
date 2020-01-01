from toolkit.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

# initialize the output frame and lock to ensure thread-safe
# exchanges of the output frames
outputFrame = None
lock = threading.Lock()

# initialize Flask
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to warmup
vs = VideoStream(usePiCamera=1).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the web page
	return render_template("index.html")

# framecount is min number of frames before building background 
def detect_motion(frameCount):
	# global references to the video stream, output frame and
	# lock variables
	global vs, outputFrame, lock

	# initialize the motion detector
	# total = frames read so far
	md = SingleMotionDetector(accumWeight=0.1)
	total = 0

	# loop over frames from vs
	while True:
		# read frame from vs, resize it
		# convert frame to grayscale and blur it
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)

		# draw timestamp on frame
		timestamp = datetime.datetime.now()
		cv2.putText(frame, timestamp.strftime(
			"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

		# if total number of frames has reached frameCount
		# continue to process the frame
		if total > frameCount:
			motion = md.detect(gray)

			# if motion was found in the frame
			if motion is not None:
				# unpack the tuple and draw the bounding box
				(thresh, (minX, minY, maxX, maxY)) = motion
				cv2.rectangle(frame, (minX, minY), (maxX, maxY),
					(0, 0, 255), 2)
		
		# update the bg and increment the total
		md.update(gray)
		total += 1

		# get the lock, set the output frame, release lock
		with lock:
			outputFrame = frame.copy()
		
def generate():
	# global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode in JPEG 
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# make sure frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

vs.stop()