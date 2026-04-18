class VisionStreamClient {
  /**
   * @param {Object} options
   * @param {string}  options.serverUrl
   * @param {string}  options.contextId
   * @param {string}  [options.userId]             - User ID (optional)
   * @param {number}  [options.interval=5]        - Minimum interval in seconds
   * @param {number}  [options.maxLongEdge=512]
   * @param {number}  [options.jpegQuality=0.8]   - JPEG quality (0.0 to 1.0)
   * @param {number}  [options.videoIdealSize=1920] - Ideal long-edge resolution for getUserMedia
   * @param {string}  [options.facingMode="user"] - "user" (front) or "environment" (back)
   * @param {HTMLVideoElement}  [options.videoElement]   - Preview element (optional)
   * @param {HTMLCanvasElement} [options.canvasElement]  - Preview element (optional)
   * @param {function} [options.pauseWhen]  - () => boolean  Pauses capture while true
   * @param {function} [options.onResult]  - (text, attentionLevel, imageUrl) => number|void  Return positive ms to cooldown
   * @param {function} [options.onError]   - (seq, error) => void
   */
  constructor({
    serverUrl,
    contextId = null,
    userId = null,
    interval = 5,
    maxLongEdge = 512,
    jpegQuality = 0.8,
    videoIdealSize = 1920,
    facingMode = "user",
    videoElement = null,
    canvasElement = null,
    pauseWhen = null,
    onResult = null,
    onError = null,
  }) {
    this.serverUrl = serverUrl;
    this.contextId = contextId;
    this.userId = userId;
    this.interval = interval;
    this.maxLongEdge = maxLongEdge;
    this.jpegQuality = jpegQuality;
    this.videoIdealSize = videoIdealSize;
    this.facingMode = facingMode;
    this.videoElement = videoElement;
    this.canvasElement = canvasElement;
    this.pauseWhen = pauseWhen;
    this.onResult = onResult;
    this.onError = onError;

    this._seq = 0;
    this._running = false;
    this._stream = null;
    this._video = null;
    this._abortController = null;

    // Internal canvas (created automatically if canvasElement is not provided)
    this._canvas = this.canvasElement || document.createElement("canvas");
    this._ctx = this._canvas.getContext("2d");
  }

  _getVideoConstraints() {
    const c = { facingMode: this.facingMode };
    if (this.videoIdealSize) {
      c.width = { ideal: this.videoIdealSize };
    }
    return c;
  }

  async _startStream() {
    if (this._stream) {
      this._stream.getTracks().forEach(t => t.stop());
    }

    this._stream = await navigator.mediaDevices.getUserMedia({
      video: this._getVideoConstraints(),
    });

    const video = this._video || this.videoElement || document.createElement("video");
    video.srcObject = this._stream;
    video.playsInline = true;
    video.style.transform = this.facingMode === "user" ? "scaleX(-1)" : "";
    await video.play();
    this._video = video;
  }

  async start() {
    if (this._running) return;
    this._running = true;

    await this._startStream();
    this._loop();
  }

  stop() {
    this._running = false;
    if (this._abortController) {
      this._abortController.abort();
      this._abortController = null;
    }
    if (this._video) {
      this._video.srcObject = null;
      this._video = null;
    }
    if (this._stream) {
      this._stream.getTracks().forEach(t => t.stop());
      this._stream = null;
    }
  }

  async switchCamera(facingMode) {
    this.facingMode = facingMode || (this.facingMode === "user" ? "environment" : "user");
    if (this._running) {
      await this._startStream();
    }
  }

  _capture() {
    const video = this._video;
    if (!video || video.readyState < video.HAVE_CURRENT_DATA) return null;

    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const longEdge = Math.max(vw, vh);

    let dw = vw, dh = vh;
    if (this.maxLongEdge && longEdge > this.maxLongEdge) {
      const scale = this.maxLongEdge / longEdge;
      dw = Math.round(vw * scale);
      dh = Math.round(vh * scale);
    }

    this._canvas.width = dw;
    this._canvas.height = dh;
    this._ctx.drawImage(video, 0, 0, dw, dh);

    this._seq++;
    return new Promise(resolve => {
      this._canvas.toBlob(blob => resolve(blob), "image/jpeg", this.jpegQuality);
    });
  }

  async capture() {
    const blob = await this._capture();
    if (!blob) return null;
    return await this._blobToDataURL(blob);
  }

  async _send(blob) {
    const form = new FormData();
    form.append("context_id", this.contextId);
    if (this.userId) form.append("user_id", this.userId);
    form.append("image", blob, "frame.jpg");

    this._abortController = new AbortController();
    const resp = await fetch(this.serverUrl, {
      method: "POST",
      body: form,
      signal: this._abortController.signal,
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    return data;
  }

  _blobToDataURL(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  async _loop() {
    while (this._running) {
      const start = performance.now();

      if (!this.contextId || (this.pauseWhen && this.pauseWhen())) {
        await new Promise(r => setTimeout(r, 500));
        continue;
      }

      const blob = await this._capture();
      if (blob) {
        try {
          const data = await this._send(blob);
          if (this.onResult) {
            const imageUrl = data.image_url || await this._blobToDataURL(blob);
            const cooldown = await this.onResult(data.text || "", data.attention_level || 0, imageUrl);
            if (cooldown > 0) {
              await new Promise(r => setTimeout(r, cooldown));
            }
          }
        } catch (e) {
          if (this.onError) this.onError(this._seq, e);
        }
      }

      const elapsed = (performance.now() - start) / 1000;
      const sleepTime = this.interval - elapsed;
      if (sleepTime > 0) {
        await new Promise(r => setTimeout(r, sleepTime * 1000));
      }
    }
  }
}
