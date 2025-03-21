import easyocr

reader = easyocr.Reader(['en'])  # 'en' for English
image_path = "C:/Users/ceogawa/Documents/428/Traffic_sign_detection/segmented_data/00000_00004_00011_png.rf.bafd7aa08ccbca5d6a09028dd395a69f_segmented.jpg"
results = reader.readtext(image_path)

for (bbox, text, prob) in results:
    print(f"Detected: {text} (Confidence: {prob:.2f})")
