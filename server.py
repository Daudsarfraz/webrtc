import asyncio
import logging
import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaStreamError
import gi
import numpy as np
import av
from av.video.frame import VideoFrame

gi.require_version("Gst", "1.0")
from gi.repository import Gst

# Initialize GStreamer
Gst.init(None)

logging.basicConfig(level=logging.DEBUG)
pcs = set()

class GStreamerTrack(VideoStreamTrack):
    kind = "video"  # Explicitly set the kind

    def __init__(self, rtsp_url):
        super().__init__()
        self.rtsp_url = rtsp_url
        self.pipeline = None
        self.queue = asyncio.Queue()
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.start_pipeline())
        # self.loop.create_task(self.on_new_sample())

        

    async def start_pipeline(self):
        print("\n============ Entered into Start pipeline Function ===================\n")
        pipeline_description = f"""
        rtspsrc location={self.rtsp_url} latency=100 protocols=tcp !
        rtph264depay !
        decodebin !
        videoconvert !
        videoscale !
        video/x-raw,format=I420,width=640,height=480 !
        appsink emit-signals=true sync=false max-buffers=1 drop=true name=appsink0
        """
        logging.debug(f"Starting GStreamer pipeline: {pipeline_description}")
        self.pipeline = Gst.parse_launch(pipeline_description)
        appsink = self.pipeline.get_by_name("appsink0")
        appsink.connect("new-sample", self.on_new_sample)
        self.pipeline.set_state(Gst.State.PLAYING)
        logging.debug("Pipeline started.")
        print("\n ===================== Done with Start pipeline =================== \n\n")


    def on_new_sample(self, sink):
        print("\n ================ Enter into On-New-Sample =============== \n")
        logging.debug("New sample received.")
        sample = sink.emit("pull-sample")

        if sample:
            buffer = sample.get_buffer()
            if buffer:
                buffer_size = buffer.get_size()
                logging.debug(f"Buffer size: {buffer_size}")  # Check buffer size

                if buffer_size == 0:
                    logging.error("Received empty buffer!")
                else:
                    logging.debug(f"Valid buffer received: {buffer_size} bytes")
                    asyncio.ensure_future(self.queue.put(buffer))
        print("\n ================ Done with On-New-Sample =============== \n")


        return Gst.FlowReturn.OK

    async def recv(self):
        print("\n ============ Enter into Recv ==================== \n\n")
        try:
            frame = await self.queue.get()
            logging.debug("Frame retrieved from queue.")

            # Convert the Gst buffer to a numpy array
            frame_data = frame.extract_dup(0, frame.get_size())
            frame_image = np.ndarray(
                (240, 320, 3),  # Ensure correct shape (height, width, channels)
                dtype=np.uint8,
                buffer=frame_data,
            )

            # Convert the frame to a format that aiortc can handle (YUV420P)
            frame_yuv = cv2.cvtColor(frame_image, cv2.COLOR_BGR2YUV_I420)

            # Convert to aiortc-compatible VideoFrame
            video_frame = VideoFrame.from_ndarray(frame_yuv, format="yuv420p")

            # Ensure correct timestamps
            video_frame.pts = None  # You can adjust the timestamp logic
            video_frame.time_base = "1/90000"  # Standard for WebRTC

            return video_frame

        except MediaStreamError:
            logging.error("MediaStreamError: Stopping pipeline.")
            self.pipeline.set_state(Gst.State.NULL)
            print("\n ======================== Recv Done ======================= \n")
            raise


async def index(request):
    html = """
    <html>
    <head>
        <title>WebRTC RTSP Stream</title>
    </head>
    <body>
        <h1>RTSP Stream to WebRTC</h1>
        <video id="video" width="640" height="480" autoplay></video>
        <script>
            const videoElement = document.getElementById("video");
            const pc = new RTCPeerConnection();
            pc.ontrack = function(event) {
                videoElement.srcObject = event.streams[0];
            };

            // Create offer
            pc.createOffer()
                .then(offer => {
                    return pc.setLocalDescription(offer);
                })
                .then(() => {
                    // Send the offer to the server
                    fetch("/offer", {
                        method: "POST",
                        body: JSON.stringify({
                            sdp: pc.localDescription.sdp,
                            type: pc.localDescription.type
                        }),
                        headers: { "Content-Type": "application/json" }
                    })
                    .then(response => response.json())
                    .then(data => {
                        return pc.setRemoteDescription(new RTCSessionDescription(data));
                    })
                    .catch(error => {
                        console.error("Error during WebRTC signaling:", error);
                    });
                })
                .catch(error => {
                    console.error("Error creating offer:", error);
                });
        </script>
    </body>
    </html>
    """
    return web.Response(content_type="text/html", text=html)

async def offer(request):
    print("\n =============== Entered into Offer function ==============\n")
    params = await request.json()
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    rtsp_url = "rtsp://admin:office2121@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"  # Replace with actual URL
    video_track = GStreamerTrack(rtsp_url)
    print("\n ================== GStreamer called ====================== \n")
    pc.addTrack(video_track) # used to send VideoTrack to other Peer 

    # Add the track event handler here inside the `offer` function
    @pc.on("track")
    async def on_track(track):
        logging.debug(f"New track received: {track.kind}")

    # Set up the remote description
    try:
        print("\n ======================== Try 1 ============================ \n")
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        await pc.setRemoteDescription(offer)
    except Exception as e:
        logging.error(f"Error setting remote description: {e}")
        raise

    # Explicitly set direction for each transceiver
    for transceiver in pc.getTransceivers():
        if transceiver.receiver and transceiver.receiver.track:
            track = transceiver.receiver.track
            if track.kind == "video":
                transceiver.direction = "sendrecv"  # Change from "sendonly" to "sendrecv"
                logging.debug(f"Set direction {transceiver.direction} for {track.kind} track.")
            else:
                logging.warning(f"Track is not video in transceiver")
        else:
            logging.warning(f"Receiver or track missing in transceiver")

    logging.debug(f"Received offer: {params['sdp']}")

    try:
        print("\n ==================== Try 2 ========================== \n")
        # Create the answer
        answer = await pc.createAnswer()
        print(f" Create Answer in TRY 2 {answer}")
        logging.debug(f"Created answer: {answer.sdp}")
        print(f"\n ============= answer.sdp in Try 2 {answer.sdp} ============== \n")
    except Exception as e:
        logging.error(f"Error creating answer: {e}")
        raise

    try:
        print("\n =================== Try 3 ============================ \n")
        # Set the local description with the answer
        logging.info(f"answer {answer}")
        print(f"Answer in TRY 3 {answer}")

        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            print(f"New ICE candidate: {candidate}")
        # Wait until ICE gathering is complete

        timeout = 10  # Time limit in seconds
        start_time = asyncio.get_event_loop().time()
        count = 0

        while pc.iceGatheringState != "complete":
            elapsed_time = asyncio.get_event_loop().time() - start_time
            if elapsed_time > timeout:
                print(f"ICE gathering timed out after {timeout} seconds, bypassing...")
                break  # Exit the loop if the timeout is reached

            print(f"Waiting for ICE gathering to complete... {count}")
            count += 1
            await asyncio.sleep(0.5)  # Wait for 0.5 seconds before checking again


        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate:
                print(f"ICE Candidate Found: {candidate}")
            else:
                print("ICE Candidate gathering completed!")


        await pc.setLocalDescription(answer)
        print(f"Answer AFTER ============== TRY 3 {pc.setLocalDescription(answer)}")

        logging.debug(f"Set local description: {answer.sdp}")
    except Exception as e:
        logging.error(f"Error setting local description: {e}")
        raise

    print("\n =============== Offer Function ==================== \n")
    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )


async def on_shutdown(app):
    print("\n =================== Entered into On-Shutdown Function ====================== \n ")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    print("\n ================ Done with On Shutdown ================== \n")
    pcs.clear()

# Function to display frames using OpenCV (cv2)
async def display_frame(rtsp_url):
    print("\n Entered in Display Frame Function::::: \n ")
    video_track = GStreamerTrack(rtsp_url)

    while True:
        frame = await video_track.recv()
        print(frame)
        if frame is not None:
            frame_resized = cv2.resize(frame.to_ndarray(format="bgr24"), (240, 320))
            cv2.imshow("RTSP Stream", frame_resized)

            # Check for 'q' key press to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

# Main Application
app = web.Application()
app.router.add_get("/", index)
app.router.add_post("/offer", offer)
app.on_shutdown.append(on_shutdown)

if __name__ == "__main__":
    print("\n ========= Starting point ============= \n")
    rtsp_url = "rtsp://admin:office2121@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"  # Replace with actual RTSP URL
    loop = asyncio.get_event_loop()
    loop.create_task(display_frame(rtsp_url))  # To display frames using OpenCV
    web.run_app(app, port=8080)


