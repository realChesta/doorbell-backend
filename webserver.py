from http.server import BaseHTTPRequestHandler, HTTPServer
import cv2

class CamServer(BaseHTTPRequestHandler):
    snapshot = None
    def do_GET(self):
        if self.path == '/snapshot.jpg':
            self.send_response(200)
            self.send_header("Content-type", "image/jpg")
            self.end_headers()
            # self.wfile.write(bytes("hi", "utf-8"))
            result, encimg = cv2.imencode('.jpg', CamServer.snapshot, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            self.wfile.write(encimg)
            return

        self.send_response(404)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(bytes("<html><head><title>Not Found</title></head>", "utf-8"))
        self.wfile.write(bytes("<body>", "utf-8"))
        self.wfile.write(bytes("<p>This page could not be found.</p>", "utf-8"))
        self.wfile.write(bytes("</body></html>", "utf-8"))
