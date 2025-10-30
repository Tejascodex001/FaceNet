import os, cv2, time
NAME = input("Enter person name (folder will be known_faces/<name>): ").strip()
save_dir = os.path.join("known_faces", NAME)
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
print("Press SPACE to capture, q to quit")

count = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow("Capture - Press SPACE to save", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        path = os.path.join(save_dir, f"{NAME}_{int(time.time())}_{count:03d}.jpg")
        cv2.imwrite(path, frame)
        print("Saved:", path)
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done. Images saved to:", save_dir)