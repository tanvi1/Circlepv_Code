from ultralytics import YOLO

    # ─── RGB Model ───────────────────────────────
    # model = YOLO("yolo26l-cls.pt")
    # results = model.train(
    #     data    = "C:\\Users\\riyasharma\\Documents\\New Project\\yolo_try\\final_RGB",
    #     epochs  = 100,
    #     batch   = 8,
    #     scale   = 0.5,
    #     fliplr  = 0.5,
    #     flipud  = 0.2,
    #     hsv_h   = 0.015,
    #     hsv_s   = 0.5,
    #     hsv_v   = 0.3,
    #     patience= 20,
    #     project = "solar_runs",
    #     name    = "rgb_model",
    #     save    = True,
    #     val     = True,
    # )

    # # ─── IR Model ────────────────────────────────
    # model = YOLO("yolo26l-cls.pt")
    # results = model.train(
    #     data    = "C:\\Users\\riyasharma\\Documents\\New Project\\yolo_try\\final_IR",
    #     epochs  = 100,
    #     batch   = 8,
    #     scale   = 0.5,
    #     fliplr  = 0.5,
    #     flipud  = 0.2,
    #     hsv_h   = 0.015,
    #     hsv_s   = 0.5,
    #     hsv_v   = 0.3,
    #     patience= 20,
    #     project = "solar_runs",
    #     name    = "ir_model",
    #     save    = True,
    #     val     = True,
    # )


from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")

results = model.train(data=r"C:\Users\riyasharma\Documents\Solar Project\detect-box\Rooftop Solar panel detection.yolo26", epochs=100, imgsz=640)