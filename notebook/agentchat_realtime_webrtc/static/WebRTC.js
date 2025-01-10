export async function init(webSocketUrl) {

    let ws
    const pc = new RTCPeerConnection();
    let dc = null;  // data connection

    async function openRTC(data) {
        const EPHEMERAL_KEY = data.client_secret.value;

        // Set up to play remote audio from the model
        const audioEl = document.createElement("audio");
        audioEl.autoplay = true;
        pc.ontrack = e => audioEl.srcObject = e.streams[0];

        // Add local audio track for microphone input in the browser
        const ms = await navigator.mediaDevices.getUserMedia({
            audio: true
        });
        pc.addTrack(ms.getTracks()[0]);

        // Set up data channel for sending and receiving events
        dc = pc.createDataChannel("oai-events");
        dc.addEventListener("message", (e) => {
            // Realtime server events appear here!
            const message = JSON.parse(e.data)
            if (message.type.includes("function")) {
                console.log("WebRTC function message", message)
                ws.send(e.data)
            }
        });

        // Start the session using the Session Description Protocol (SDP)
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        const baseUrl = "https://api.openai.com/v1/realtime";
        const model = data.model;
        const sdpResponse = await fetch(`${baseUrl}?model=${model}`, {
            method: "POST",
            body: offer.sdp,
            headers: {
                Authorization: `Bearer ${EPHEMERAL_KEY}`,
                "Content-Type": "application/sdp"
            },
        });

        const answer = {
            type: "answer",
            sdp: await sdpResponse.text(),
        };
        await pc.setRemoteDescription(answer);
        console.log("Connected to OpenAI WebRTC")
    }

    ws = new WebSocket(webSocketUrl);

    ws.onopen = event => {
        console.log("web socket opened")
    }

    ws.onmessage = async event => {
        const message = JSON.parse(event.data)
        console.info("Received Message from AG2 backend", message)
        const type = message.type
        if (type == "ag2.init") {
            await openRTC(message.config)
            return
        }
        const messageJSON = JSON.stringify(message)
        if (dc) {
            dc.send(messageJSON)
        } else {
            console.log("DC not ready yet", message)
        }
    }
}
