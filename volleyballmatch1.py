import cv2 as cv
import numpy as np
import math

video_path = "C:\\Users\\shiva\\Downloads\\volleyball_match.mp4"
cap = cv.VideoCapture(video_path)

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))

# output video format 
ball_tracking_output = "volleyball_tracking.mp4"
fourcc = cv.VideoWriter_fourcc(*'mp4v')
writer_ball = cv.VideoWriter(ball_tracking_output, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # color range for the ball
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([40, 255, 255])
    
    mask = cv.inRange(hsv, yellow_lower, yellow_upper)
    
    # thresholding and edge detection using canny and then masking
    edges = cv.Canny(mask, 30, 100)
    kernel = np.ones((7,7), np.uint8)
    mask = cv.dilate(edges, kernel, iterations=3)
    
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    best_match = None
    max_circularity = 0

    for cnt in contours:
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if 150 < area < 800 and 0.5 < circularity < 1.3:
            if circularity > max_circularity:
                max_circularity = circularity
                best_match = cnt

    if best_match is not None:
        x, y, w, h = cv.boundingRect(best_match)
        center = (x + w // 2, y + h // 2)
        cv.circle(frame, center, 12, (0, 255, 0), 3)
        cv.putText(frame, 'Ball', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    writer_ball.write(frame)
    cv.imshow("Ball Tracking", frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
writer_ball.release()
cv.destroyAllWindows()

print(f"Ball tracking video saved as {ball_tracking_output}")

# detect the players
cap = cv.VideoCapture(video_path)
player_detection_output = "player_detection.mp4"
writer_players = cv.VideoWriter(player_detection_output, fourcc, fps, (frame_width, frame_height))

# to store number of players 
red_team_history = []
yellow_team_history = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # colors for the team
    red_lower = np.array([170, 100, 50])
    red_upper = np.array([180, 255, 255])
    yellow_lower = np.array([15, 100, 100])
    yellow_upper = np.array([35, 255, 255])
    
    red_mask = cv.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv.inRange(hsv, yellow_lower, yellow_upper)
    
    # morphology as haar cascade wasnt accurate to detect the players
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)
    yellow_mask = cv.morphologyEx(yellow_mask, cv.MORPH_CLOSE, kernel)
    red_mask = cv.dilate(red_mask, kernel, iterations=3)
    yellow_mask = cv.dilate(yellow_mask, kernel, iterations=3)
    
    red_contours, _ = cv.findContours(red_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    yellow_contours, _ = cv.findContours(yellow_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    def detect_players(contour):
        area = cv.contourArea(contour)
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = w / float(h)
        return 800 < area < 7000 and 0.2 < aspect_ratio < 1.5

    red_players = sum(1 for c in red_contours if detect_players(c))
    yellow_players = sum(1 for c in yellow_contours if detect_players(c))
    red_team_history.append(red_players)
    yellow_team_history.append(yellow_players)

    cv.putText(frame, f"Red Team: {red_players}", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv.putText(frame, f"Yellow Team: {yellow_players}", (50, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    writer_players.write(frame)
    cv.imshow("Player Detection", frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
writer_players.release()
cv.destroyAllWindows()
# take the average values and ceiling value for the number of players
avg_red_players = math.ceil(sum(red_team_history) / len(red_team_history)) if red_team_history else 0
avg_yellow_players = math.ceil(sum(yellow_team_history) / len(yellow_team_history)) if yellow_team_history else 0

print(f"Average Red Team Players: {avg_red_players}")
print(f"Average Yellow Team Players: {avg_yellow_players}")
print(f"Player detection video saved as {player_detection_output}")
