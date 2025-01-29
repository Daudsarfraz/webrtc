import cv2

def test_rtsp_stream(rtsp_url, width=320, height=240):
    # Open RTSP stream using OpenCV
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return

    while True:
        # Read a frame from the RTSP stream
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to retrieve frame.")
            break

        # Resize the frame to the desired width and height
        frame_resized = cv2.resize(frame, (width, height))

        # Display the resized frame using OpenCV
        cv2.imshow("RTSP Stream", frame_resized)

        # Press 'q' to quit the stream display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# RTSP URL to test
rtsp_url = "rtsp://admin:office2121@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"  # Replace with your RTSP URL

# Run the test with a resize to 640x480 (240, 320, 3),  # Ensure correct shape (height, width, channels)

test_rtsp_stream(rtsp_url, width=320, height=240)
