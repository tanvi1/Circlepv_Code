from ultralytics import YOLO

# 1. Load your custom trained model weights
model = YOLO("bestn.pt")
img="C:\\Users\\riyasharma\\Documents\\Solar Project\\solar-panel-detection\\solar-panels\\1.png"
# 2. Run testing/inference on your custom image
results = model.predict(source=img, save=True, conf=0.25, classes=[1])

# 3. View results (optional: opens window if running locally)
for result in results:
    print(result.boxes.cls)   # Prints detected class IDs
    print(result.boxes.conf) 
