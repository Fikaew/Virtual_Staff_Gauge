import cv2
import numpy as np
import tensorflow as tf
import math
from fastapi import FastAPI

get_ref_1 = False
get_ref_2 = False

ref_edge = None
ref_edge_rgb = None

ref_2_edge = None
ref_2_edge_rgb = None

edge = None
edge_rgb = None

line_start_point = None
line_end_point = None
line_BGR_color = (0, 255, 255) #Yellow
line_thickness = 2
is_line = False

collision_ref_edge = None
collision_ref_point = None
collision_ref_2_edge = None
collision_ref_2_point = None
collision_edge = None
collision_point = None

dot_radius = 3
dot_color = (0, 0, 255)

ref_magnitude = None
real_ref_height = None
real_ref_2_height = None
estimated_dis = None
estimated_height = None

def read_image(image):
    image = tf.image.resize(images=image, size=[tf.shape(image)[0], tf.shape(image)[1]])
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

def prediction_mask(model, image_path):
    image = read_image(image_path)
    predicted_mask = model.predict(tf.expand_dims(image, axis=0), verbose=0)[0]
    return predicted_mask

def cvtImage2Gray(image, threshold): #convert mask to Grayscale
    image = np.where(image >= threshold, 1, 0)
    image = image.astype(np.uint8) * 255
    return image

def process_frame(model, frame, width, height):
    frame = cv2.resize(frame, (width, height))
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predicted_mask = prediction_mask(model, frame)
    gray_mask = cvtImage2Gray(predicted_mask, 0.5)
    gray_mask = cv2.resize(gray_mask, (width, height))
    edge = cv2.Canny(gray_mask, 0, 255)
    edge_rgb = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    edge_rgb[np.where(edge_rgb[:, :, 0])] = (255, 0, 255) #Magenta

def get_line_pos(event, x, y, flags, param):
    global line_start_point, line_end_point, is_line, width, height
    if event == cv2.EVENT_LBUTTONDOWN:
        line_start_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if x > width:
            x = width - 1
        elif x < 0:
            x = 0
        if y > height:
            y = height - 1
        elif y < 0:
            y = 0
        line_end_point = (x, y)
        is_line = True

def draw_line(frame, start_point, end_point):
    if line_start_point is not None and line_end_point is not None:
        cv2.line(frame, start_point, end_point, line_BGR_color, line_thickness)

def cal_distance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    distance = math.sqrt(dx ** 2 + dy **2)
    return distance

def line_edge_collision(start_point, end_point, edge_mask):
    x0, y0 = start_point
    x1, y1 = end_point

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    err = dx - dy

    while True:
        if edge_mask[y0, x0] != 0:
            return True, (x0, y0)

        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return False, None

model = tf.keras.models.load_model("Main/model.h5")

video_path = "Main/video/img2vdo2.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("Edge")
cv2.setMouseCallback("Edge", get_line_pos)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    key = cv2.waitKey(100)
    if key == ord('q'):
        break
    
    frame = cv2.resize(frame, (width, height))
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predicted_mask = prediction_mask(model, frame)
    gray_mask = cvtImage2Gray(predicted_mask, 0.5)
    gray_mask = cv2.resize(gray_mask, (width, height))
    edge = cv2.Canny(gray_mask, 0, 255)
    edge_rgb = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    edge_rgb[np.where(edge_rgb[:, :, 0])] = (255, 0, 255) #Magenta

    if key == ord('a'):
    # if get_ref_1: #save current frame as ref
        real_ref_height = float(input("Insert Ref Height: "))
        ref_mask = gray_mask.copy()
        ref_edge = edge.copy()
        ref_edge_rgb = edge_rgb.copy()
        ref_edge_rgb[np.where(edge_rgb[:, :, 0])] = (255, 255, 0) #Cyan
        ref_frame = cv2.bitwise_or(frame, ref_edge_rgb)
        cv2.imshow("Edge", ref_frame)
        get_ref_1 = False
    elif key == ord('b'):
    # elif get_ref_2: #save current frame as ref 2
        real_ref_2_height = float(input("Insert Ref 2 Height: "))
        ref_2_mask = gray_mask.copy()
        ref_2_edge = edge.copy()
        ref_2_edge_rgb = edge_rgb.copy()
        ref_2_edge_rgb[np.where(edge_rgb[:, :, 0])] = (0, 255, 0) #Green
        ref_2_frame = cv2.bitwise_or(frame, ref_2_edge_rgb)
        cv2.imshow("Edge", ref_2_frame)
        get_ref_2 = False

    main_frame = frame
    main_frame = cv2.bitwise_or(main_frame, edge_rgb)
    cv2.imshow("Edge", main_frame)
    draw_line(main_frame, line_start_point, line_end_point)
    if is_line:
        collision_edge, collision_point = line_edge_collision(line_start_point, line_end_point, edge)
        cv2.circle(main_frame, collision_point, dot_radius, dot_color, -1)
    if ref_edge_rgb is not None and is_line:
        # main_frame = cv2.bitwise_or(main_frame, ref_edge_rgb)
        collision_ref_edge, collision_ref_point = line_edge_collision(line_start_point, line_end_point, ref_edge)
        cv2.circle(main_frame, collision_ref_point, dot_radius, (255, 255, 0), -1)
    if ref_2_edge_rgb is not None and is_line:
        # main_frame = cv2.bitwise_or(main_frame, ref_2_edge_rgb)
        collision_ref_2_edge, collision_ref_2_point = line_edge_collision(line_start_point, line_end_point, ref_2_edge)

    if collision_ref_edge and collision_ref_2_edge:
        ref_distance_magnitude = cal_distance(collision_ref_point[0], collision_ref_point[1], collision_ref_2_point[0], collision_ref_2_point[1])
        if collision_edge and real_ref_height is not None and real_ref_2_height is not None and ref_distance_magnitude != 0:
            temp_dis = cal_distance(collision_ref_point[0], collision_ref_point[1], collision_point[0], collision_point[1])
            text = ""
            estimate_dis = (temp_dis * abs(real_ref_height - real_ref_2_height)) / ref_distance_magnitude
            if collision_point[1] > collision_ref_point[1]:
                text = "-"
                print(collision_point[1], collision_ref_point[1])
                estimate_height = real_ref_height - estimate_dis
            else:
                text = "+"
                estimate_height = real_ref_height + estimate_dis
            text += str(estimate_dis)
            cv2.putText(main_frame, text, (collision_point[0] + 15, collision_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(main_frame, str(estimate_height), (collision_point[0] + 15, collision_point[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            print("estimate height: ", estimate_height)
        elif ref_distance_magnitude == 0 and collision_point is not None:
            cv2.putText(main_frame, "please select another ref point", (collision_point[0] + 15, collision_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow("Edge", main_frame)
    output_path = f"Main/oot/backend/static/frame_{frame_count:03d}.png"
    cv2.imwrite(output_path, main_frame)
    if frame_count < 100:
       frame_count += 1
    else:
        frame_count = 0

cap.release()
cv2.destroyAllWindows()
