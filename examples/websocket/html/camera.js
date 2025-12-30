class Camera {
    /**
     * @param {Object} options
     * @param {HTMLVideoElement} options.videoElement - Video element for camera stream
     * @param {HTMLCanvasElement} options.canvasElement - Canvas element for capturing frames
     * @param {Function} options.onCapture - Callback function that receives captured image as data URL
     * @param {string} [options.imageFormat='image/jpeg'] - Image format for capture
     * @param {number} [options.imageQuality=0.8] - Image quality (0-1) for JPEG/WebP
     */
    constructor({ videoElement, canvasElement, onCapture, imageFormat = 'image/jpeg', imageQuality = 0.8 }) {
        this.video = videoElement;
        this.canvas = canvasElement;
        this.onCapture = onCapture;
        this.imageFormat = imageFormat;
        this.imageQuality = imageQuality;
        this.stream = null;
        this.isInitialized = false;
    }

    /**
     * Initialize the camera and request permissions
     * @returns {Promise<boolean>} True if initialization succeeded
     */
    async init() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: 'user',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            });
            this.video.srcObject = this.stream;
            await this.video.play();
            this.isInitialized = true;
            console.log('Camera initialized successfully');
            return true;
        } catch (error) {
            console.error('Failed to initialize camera:', error);
            this.isInitialized = false;
            return false;
        }
    }

    /**
     * Capture a photo and invoke the onCapture callback
     * @returns {Promise<string|null>} Data URL or null if failed
     */
    async capture() {
        if (!this.isInitialized) {
            console.warn('Camera not initialized. Call init() first.');
            return null;
        }

        try {
            // Set canvas size to match video
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;

            // Draw current video frame to canvas
            const ctx = this.canvas.getContext('2d');
            ctx.drawImage(this.video, 0, 0);

            // Get data URL
            const dataUrl = this.canvas.toDataURL(this.imageFormat, this.imageQuality);

            // Invoke callback
            if (this.onCapture) {
                this.onCapture(dataUrl);
            }

            return dataUrl;
        } catch (error) {
            console.error('Failed to capture image:', error);
            return null;
        }
    }

    /**
     * Stop the camera stream and release resources
     */
    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        this.video.srcObject = null;
        this.isInitialized = false;
        console.log('Camera stopped');
    }
}
