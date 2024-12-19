import { Audio } from './Audio.js';

// Create an instance of AudioPlayer with the WebSocket URL
const audio = new Audio(socketUrl);
// Start receiving and playing audio
audio.start();