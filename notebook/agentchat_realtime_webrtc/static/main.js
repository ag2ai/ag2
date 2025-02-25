const main = async () => {
    const eConnecting = document.getElementById("connecting")
    const eConnected = document.getElementById("connected")
    const eDisconnected = document.getElementById("disconnected")

    eConnecting.style.display = "block"
    eConnected.style.display = "none"
    eDisconnected.style.display = "none"
    const webRTC = new ag2client.WebRTC(socketUrl)
    webRTC.onDisconnect = () => {
        eDisconnected.style.display = "block"
        eConnected.style.display = "none"
        eConnecting.style.display = "none"
    }
    await webRTC.connect();
    eConnecting.style.display = "none"
    eConnected.style.display = "block"
}

main()
