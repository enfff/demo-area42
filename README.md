# Quickstart

    python3 -m venv venv
    source venv/bin/activate
    pip install requests opencv-python numpy flask ultralytics flask

<!-- Non funziona? Prova pip install -r requirements.txt -->

Adesso esegui il codice con

    python3 demo_spot.py

# Cosa fa sto script?

Accetta un video da SPOT, esegue l'inferenza su ogni fotogramma, e se `DETECTIONS_THRESHOLD` fotogrammi di fila mostrano una fence bucata con una confidence superiore a `YOLO_CONFIDENCE_THRESHOLD`, allora viene spedita una task di delivery (`delivery_payload_template`) con una POST verso l'indirizzo `POST_URL`