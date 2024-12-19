// AudioPlayer.js

export class AudioPlayer {
    constructor(webSocketUrl) {
      this.webSocketUrl = webSocketUrl;
      this.socket = null;
      this.audioContext = null;
      this.sourceNode = null;
      this.bufferQueue = [];  // Queue to store audio buffers
      this.isPlaying = false; // Flag to check if audio is playing
    }
  
    // Initialize WebSocket and start receiving audio data
    async start() {
      try {
        // Initialize WebSocket connection
        this.socket = new WebSocket(this.webSocketUrl);
  
        this.socket.onopen = () => {
          console.log("WebSocket connected.");
        };
  
        this.socket.onclose = () => {
          console.log("WebSocket disconnected.");
        };
  
        this.socket.onmessage = async (event) => {
          // Ensure the data is an ArrayBuffer, if it's a Blob, convert it
          const pcmData = event.data instanceof ArrayBuffer ? event.data : await event.data.arrayBuffer();
          this.queuePcmData(pcmData);  // Push the received data into the buffer queue
          if (!this.isPlaying) {
            this.playFromQueue();  // Start playing if not already playing
          }
        };
  
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        console.log("Audio player initialized.");
      } catch (err) {
        console.error("Error initializing audio player:", err);
      }
    }
  
    // Stop receiving and playing audio
    stop() {
      if (this.socket) {
        this.socket.close();
      }
      if (this.audioContext) {
        this.audioContext.close();
      }
      console.log("Audio player stopped.");
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
      const source = this.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.audioContext.destination);
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
      const audioBuffer = this.audioContext.createBuffer(1, audioData.length, this.audioContext.sampleRate);
      audioBuffer.getChannelData(0).set(audioData);
  
      return audioBuffer;
    }
  }
  