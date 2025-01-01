import cv2
import numpy as np
from scipy.spatial import distance as dist, distance


class Robot:
    def __init__(self, pos=None, team='-no team!-', ID='-no ID!-'):
        self.pos = pos if pos is not None else []
        self.team = team
        self.ID = ID
        self.circles = []

    def add_marking(self, circle=None):
        if circle is None:
            circle = [0, 0, [0, 0, 0]]
        self.circles.append(circle)


class Ball:
    def __init__(self, pos=None):
        self.pos = pos if pos is not None else []


ball = Ball()
robotList = []
robotMarks = []

def Color_Detection(blue, green, red):
    if blue > 220 and green < 50 and red < 50:
        return 'Blue'
    if blue < 50 and green > 200 and red > 200:
        return 'Yellow'
    if blue > 200 and green < 50 and red > 200:
        return 'Purple'
    if blue < 50 and green > 220 and red < 50:
        return 'Green'
    if blue < 50 and green < 200 and red > 180:
        return 'Orange'
    return 'Unidentified'


def IdentifyCircles(img, circle):
    global ball

    x, y = int(circle[0]), int(circle[1])
    blue, green, red = img[y, x, 0], img[y, x, 0], img[y, x, 0]
    color = Color_Detection(blue, green, red)

    if color == 'Blue' or color == 'Yellow':
        robotList.append(Robot([x, y], color))
    elif color == 'Green' or color == 'Purple':
        robotMarks.append([x, y, color])
        print('ROBOT FOUND')
    elif color == 'Orange':
        ball.pos = [x, y]
        print(f"Ball found at ({x}, {y})")


def assignIDmarks():
    if robotList is not None:
        for idx, robot in enumerate(robotList):
            distances = []

            for i, mark in enumerate(robotMarks):
                mark_dist = distance.euclidean(mark[:2], robot.pos)
                distances.append((i, mark_dist))
            distances.sort(key=lambda x: x[1])
            closest_marks_indices = [i for i, _ in distances[:4]]
            robot.circles = [robotMarks[i] for i in closest_marks_indices]
            robot.ID = idx + 1


def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist=20, param1=50, param2=14, minRadius=20, maxRadius=59)
    return circles


def annotate_image(img):
    for robot in robotList:
        team_color = "B" if robot.team == 'Blue' else "Y"
        cv2.putText(img, f'{team_color}', (robot.pos[0] + 20, robot.pos[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, .75,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f'ID{robot.ID}', (robot.pos[0] + 20, robot.pos[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, .75,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f'{robot.pos}', (robot.pos[0] + 20, robot.pos[1]), cv2.FONT_HERSHEY_SIMPLEX, .75,
                    (255, 255, 255), 2, cv2.LINE_AA)

    if ball.pos:
        cv2.putText(img, f'Ball {ball.pos}', (ball.pos[0] + 20, ball.pos[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (ball.pos[0], ball.pos[1]), 10, (0, 165, 255), -1)  # Orange color for the ball


# Main function
def main():
    global robotList, robotMarks
    global ball

    # Open the video file
    video_path = "Assets/Video/Test2.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open video file: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Initialize globals for each frame
        robotList = []
        robotMarks = []
        ball = Ball()  # Ensure ball is always an instance of Ball

        # Detect circles in the frame
        circles = detect_circles(frame)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                IdentifyCircles(frame, circle)
                cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                cv2.circle(frame, (circle[0], circle[1]), 2, (0, 0, 255), 3)

            assignIDmarks()

            for robot in robotList:
                print(f'There is a {robot.team} robot with these ID circles:')
                for mark in robot.circles:
                    print(mark)

            if ball.pos:
                print(f'Ball found at {ball.pos}')

            for robot in robotList:
                if robot.pos:
                    cv2.circle(frame, (robot.pos[0], robot.pos[1]), 10, (0, 0, 0), 5)
                    for mark in robot.circles:
                        cv2.circle(frame, (mark[0], mark[1]), 10, (0, 0, 0), 5)

        else:
            print("No circles detected")

        annotate_image(frame)
        cv2.imshow("Annotated Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
