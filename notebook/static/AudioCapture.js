export class AudioCapture {
    constructor(webSocketUrl) {
      this.webSocketUrl = webSocketUrl;
      this.socket = null;
      this.audioContext = null;
      this.processorNode = null;
      this.stream = null;
      this.bufferSize = 8192;  // Define the buffer size for capturing chunks
    }
  
    // Initialize WebSocket and start capturing audio
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
  
        // Get user media (microphone access)
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this.stream = stream;
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
  
        // Create an AudioNode to capture the microphone stream
        const sourceNode = this.audioContext.createMediaStreamSource(stream);
  
        // Create a ScriptProcessorNode (or AudioWorkletProcessor for better performance)
        this.processorNode = this.audioContext.createScriptProcessor(this.bufferSize, 1, 1);
        
        // Process audio data when available
        this.processorNode.onaudioprocess = (event) => {
          const inputBuffer = event.inputBuffer;
          const outputBuffer = event.outputBuffer;
  
          // Extract PCM 16-bit data from input buffer (mono channel)
          const audioData = this.extractPcm16Data(inputBuffer);
          
          // Send the PCM data over the WebSocket
          if (this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(audioData);
          }
        };
  
        // Connect the source node to the processor node and the processor node to the destination (speakers)
        sourceNode.connect(this.processorNode);
        this.processorNode.connect(this.audioContext.destination);
  
        console.log("Audio capture started.");
      } catch (err) {
        console.error("Error capturing audio:", err);
      }
    }
  
    // Stop capturing audio and close the WebSocket connection
    stop() {
      if (this.processorNode) {
        this.processorNode.disconnect();
      }
      if (this.audioContext) {
        this.audioContext.close();
      }
      if (this.socket) {
        this.socket.close();
      }
      if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop());
      }
  
      console.log("Audio capture stopped.");
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
  