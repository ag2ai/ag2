// Audio.js

export class Audio {
    constructor(webSocketUrl) {
        this.webSocketUrl = webSocketUrl;
        this.socket = null;
        // audio out
        this.outAudioContext = null;
        this.sourceNode = null;
        this.bufferQueue = [];  // Queue to store audio buffers
        this.isPlaying = false; // Flag to check if audio is playing
        // audio in
        this.inAudioContext = null;
        this.processorNode = null;
        this.stream = null;
        this.bufferSize = 8192;  // Define the buffer size for capturing chunks
    }

    // Initialize WebSocket and start receiving audio data
    async start() {
        try {
            // Initialize WebSocket connection
            this.socket = new WebSocket(this.webSocketUrl);

            this.socket.onopen = () => {
                console.log("WebSocket connected.");
                const sessionStarted = {
                    event: "start",
                    start: {
                        streamSid: crypto.randomUUID(),
                    }
                }
                this.socket.send(JSON.stringify(sessionStarted))
                console.log("sent session start")
                };

            this.socket.onclose = () => {
                console.log("WebSocket disconnected.");
            };

            this.socket.onmessage = async (event) => {
                console.log("Received web socket message")
                const message = JSON.parse(event.data)
                if (message.event == "media") {
                    const bufferString = atob(message.media.payload); // Decode base64 to binary string
                    const byteArray = new Uint8Array(bufferString.length);
                    for (let i = 0; i < bufferString.length; i++) {
                      byteArray[i] = bufferString.charCodeAt(i); //Create a byte array
                    }
                    //const payload = base64.decode(message.media.payload)
                    // Ensure the data is an ArrayBuffer, if it's a Blob, convert it
                    //const pcmData = event.data instanceof ArrayBuffer ? event.data : await event.data.arrayBuffer();
                    //

                    this.queuePcmData(byteArray.buffer);  // Push the received data into the buffer queue
                    if (!this.isPlaying) {
                            this.playFromQueue();  // Start playing if not already playing
                    }
                }
            };
            this.outAudioContext = new (window.AudioContext || window.webkitAudioContext)();
            console.log("Audio player initialized.");

            // audio in
            // Get user media (microphone access)

            const stream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate:24000}  });
            this.stream = stream;
            this.inAudioContext = new AudioContext({ sampleRate: 24000 });

            // Create an AudioNode to capture the microphone stream
            const sourceNode = this.inAudioContext.createMediaStreamSource(stream);

            // Create a ScriptProcessorNode (or AudioWorkletProcessor for better performance)
            this.processorNode = this.inAudioContext.createScriptProcessor(this.bufferSize, 1, 1);

            // Process audio data when available
            this.processorNode.onaudioprocess = (event) => {
                const inputBuffer = event.inputBuffer;

                // Extract PCM 16-bit data from input buffer (mono channel)
                const audioData = this.extractPcm16Data(inputBuffer);
                const byteArray = new Uint8Array(audioData); // Create a Uint8Array view
                const bufferString = String.fromCharCode(...byteArray); // convert each byte of the buffer to a character
                const audioBase64String = btoa(bufferString); // Apply base64
                // Send the PCM data over the WebSocket
                if (this.socket.readyState === WebSocket.OPEN) {
                    const audioMessage = {
                        'event': "media",
                        'media': {
                            'timestamp': Date.now(),
                            'payload': audioBase64String
                        }
                    }
                    this.socket.send(JSON.stringify(audioMessage));
                }
            };

            // Connect the source node to the processor node and the processor node to the destination (speakers)
            sourceNode.connect(this.processorNode);
            this.processorNode.connect(this.inAudioContext.destination);
            console.log("Audio capture started.");
        } catch (err) {
            console.error("Error initializing audio player:", err);
        }
    }

    // Stop receiving and playing audio
    stop() {
        this.stop_out()
        this.stop_in()
    }

    stop_out() {
        if (this.socket) {
            this.socket.close();
        }
        if (this.outAudioContext) {
            this.outAudioContext.close();
        }
        console.log("Audio player stopped.");
    }

    stop_in() {
        if (this.processorNode) {
            this.processorNode.disconnect();
        }
        if (this.inAudioContext) {
            this.inAudioContext.close();
        }
        if (this.socket) {
            this.socket.close();
        }
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
        }
        console.log("Audio capture stopped.");
    }

    // Queue PCM data for later playback
    queuePcmData(pcmData) {
        this.bufferQueue.push(pcmData);
    }

    // Play audio from the queue
    async playFromQueue() {
        if (this.bufferQueue.length === 0) {
            this.isPlaying = false; // No more data to play
            return;
        }

        this.isPlaying = true;
        const pcmData = this.bufferQueue.shift();  // Get the next chunk from the queue

        // Convert PCM 16-bit data to ArrayBuffer
        const audioBuffer = await this.decodePcm16Data(pcmData);

        // Create an audio source and play it
        const source = this.outAudioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.outAudioContext.destination);
        source.onended = () => {
            // Play the next chunk after the current one ends
            this.playFromQueue();
        };
        source.start();
    }

    // Decode PCM 16-bit data into AudioBuffer
    async decodePcm16Data(pcmData) {
        const audioData = new Float32Array(pcmData.byteLength / 2);

        // Convert PCM 16-bit to Float32Array
        const dataView = new DataView(pcmData);
        for (let i = 0; i < audioData.length; i++) {
            const pcm16 = dataView.getInt16(i * 2, true); // true means little-endian
            audioData[i] = pcm16 / 32768;  // Convert to normalized float (-1 to 1)
        }

        // Create an audio buffer from the Float32Array
        const audioBuffer = this.outAudioContext.createBuffer(1, audioData.length, 24000);
        audioBuffer.getChannelData(0).set(audioData);

        return audioBuffer;
    }

    // Convert audio buffer to PCM 16-bit data
    extractPcm16Data(buffer) {
        const sampleRate = buffer.sampleRate;
        const length = buffer.length;
        const pcmData = new Int16Array(length);

        // Convert the float samples to PCM 16-bit (scaled between -32768 and 32767)
        for (let i = 0; i < length; i++) {
            pcmData[i] = Math.max(-32768, Math.min(32767, buffer.getChannelData(0)[i] * 32767));
        }

        // Convert Int16Array to a binary buffer (ArrayBuffer)
        const pcmBuffer = new ArrayBuffer(pcmData.length * 2); // 2 bytes per sample
        const pcmView = new DataView(pcmBuffer);

        for (let i = 0; i < pcmData.length; i++) {
            pcmView.setInt16(i * 2, pcmData[i], true); // true means little-endian
        }

        return pcmBuffer;
    }

  }
