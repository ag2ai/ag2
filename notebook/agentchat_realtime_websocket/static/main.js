// Create an instance of AudioPlayer with the WebSocket URL
console.log(ag2client);
const audio = new ag2client.WebsocketAudio(socketUrl);
// Start receiving and playing audio
audio.start();
