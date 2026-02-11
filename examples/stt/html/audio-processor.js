/**
 * AudioWorkletProcessor for capturing microphone audio
 * Runs in a separate audio thread for better performance
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this.bufferSize = options.processorOptions?.bufferSize || 512;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) {
            return true;
        }

        const inputChannel = input[0];

        for (let i = 0; i < inputChannel.length; i++) {
            this.buffer[this.bufferIndex++] = inputChannel[i];

            if (this.bufferIndex >= this.bufferSize) {
                // Calculate RMS
                let sum = 0;
                for (let j = 0; j < this.bufferSize; j++) {
                    sum += this.buffer[j] * this.buffer[j];
                }
                const rms = Math.sqrt(sum / this.bufferSize);

                // Send buffer to main thread
                this.port.postMessage({
                    audioData: this.buffer.slice(),
                    rms: rms
                });

                this.bufferIndex = 0;
            }
        }

        return true;
    }
}

registerProcessor("audio-capture-processor", AudioCaptureProcessor);
