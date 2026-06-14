# src/dashboard/server.py
import http.server
import socketserver
import os
import sys

PORT = 8000

# Change working directory to the project root to serve files correctly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.chdir(PROJECT_ROOT)

class MonopolyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Disable caching for active development
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def run_server():
    # Use standard TCPServer with reuse_address to avoid port conflicts
    socketserver.TCPServer.allow_reuse_address = True
    handler = MonopolyHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            print("======================================================================")
            print(f"🎩 JAX Monopoly Dashboard running at: http://localhost:{PORT}/src/dashboard/index.html")
            print("Press Ctrl+C to stop the server.")
            print("======================================================================")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_server()
