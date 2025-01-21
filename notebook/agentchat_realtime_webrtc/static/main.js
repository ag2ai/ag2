import { init } from './WebRTC.js';

const main = async () => {
    const eConnecting = document.getElementById("connecting")
    const eConnected = document.getElementById("connected")
    eConnecting.style.display = "block"
    eConnected.style.display = "none"
    await ag2Connect(socketUrl);
    eConnecting.style.display = "none"
    eConnected.style.display = "block"
}

main()
