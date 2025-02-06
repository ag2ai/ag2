const main = async () => {
    const eConnecting = document.getElementById("connecting")
    const eConnected = document.getElementById("connected")
    eConnecting.style.display = "block"
    eConnected.style.display = "none"
    const webRTC = new ag2client.WebRTC(socketUrl)
    await webRTC.connect();
    eConnecting.style.display = "none"
    eConnected.style.display = "block"
}

main()
