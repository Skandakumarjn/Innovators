import streamlit as st
import json
import os
from datetime import datetime, timedelta
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np
import pygame
import requests
import threading
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords

USER_DATA_FILE = 'users.json'
SLEEP_DATA_FILE = 'sleep_data.json'

def load_user_data():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, 'r') as file:
            return json.load(file)
    else:
        return {}

def load_sleep_data():
    if os.path.exists(SLEEP_DATA_FILE):
        with open(SLEEP_DATA_FILE, 'r') as file:
            return json.load(file)
    else:
        return {}

def save_user_data(users):
    with open(USER_DATA_FILE, 'w') as file:
        json.dump(users, file)

def save_sleep_data(sleep_data):
    with open(SLEEP_DATA_FILE, 'w') as file:
        json.dump(sleep_data, file)

if 'users' not in st.session_state:
    st.session_state.users = load_user_data()
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'sleep_data' not in st.session_state:
    st.session_state.sleep_data = load_sleep_data()

pygame.mixer.init()
alert_sound = pygame.mixer.Sound('default_alert.wav')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

frame_width = 1024
frame_height = 576
EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.79
ALERT_DURATION_THRESHOLD = 1.5
alert_playing = False
eye_closed_start_time = None


def get_time_of_day():
    """Returns the time of day as a string."""
    current_time = datetime.now().time()
    if current_time < datetime.strptime("12:00", "%H:%M").time():
        return "Morning"
    elif current_time < datetime.strptime("17:00", "%H:%M").time():
        return "Afternoon"
    elif current_time < datetime.strptime("20:00", "%H:%M").time():
        return "Evening"
    else:
        return "Night"

def create_account():
    st.subheader("Create Account")
    username = st.text_input("Choose a Username", key="create_username")
    password = st.text_input("Choose a Password", type="password", key="create_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")

    if st.button("Create Account"):
        if username in st.session_state.users:
            st.warning("Username already exists. Please choose a different one.")
        elif password != confirm_password:
            st.warning("Passwords do not match.")
        else:
            st.session_state.users[username] = password
            st.session_state.sleep_data[username] = []
            save_user_data(st.session_state.users)  # Save user data to file
            save_sleep_data(st.session_state.sleep_data)  # Save sleep data to file
            st.success("Account created successfully!")
            st.session_state.create_username = ""
            st.session_state.create_password = ""
            st.session_state.confirm_password = ""

def login():
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.success("Logged in successfully!")
            st.session_state.logged_in = True
            st.session_state.current_user = username
            if username not in st.session_state.sleep_data:
                st.session_state.sleep_data[username] = []
                save_sleep_data(st.session_state.sleep_data)  # Save sleep data to file
        else:
            st.warning("Invalid username or password.")

def enter_sleep_times():
    st.subheader("Enter Sleep Time")

    hours = [f"{hour:02}" for hour in range(1, 13)]
    periods = ["AM", "PM"]

    sleep_col1, sleep_col2 = st.columns(2)
    wake_col1, wake_col2 = st.columns(2)

    with sleep_col1:
        sleep_hour = st.selectbox("Sleep Hour", hours, key="sleep_hour")
    with sleep_col2:
        sleep_period = st.selectbox("Sleep Period", periods, key="sleep_period")

    with wake_col1:
        wake_hour = st.selectbox("Wake Hour", hours, key="wake_hour")
    with wake_col2:
        wake_period = st.selectbox("Wake Period", periods, key="wake_period")

    if st.button("Submit"):
        if st.session_state.current_user:
            try:
                sleep_time_str = f"{sleep_hour} {sleep_period}"
                wake_time_str = f"{wake_hour} {wake_period}"
                sleep_time = datetime.strptime(sleep_time_str, "%I %p")
                wake_time = datetime.strptime(wake_time_str, "%I %p")
                if wake_time < sleep_time:
                    wake_time += timedelta(days=1)
                sleep_duration = wake_time - sleep_time
                sleep_hours = sleep_duration.total_seconds() / 3600
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.sleep_data[st.session_state.current_user].append((sleep_hours, timestamp))
                save_sleep_data(st.session_state.sleep_data)  # Save sleep data to file
                st.success(
                    f"You slept from {sleep_time.strftime('%I %p')} to {wake_time.strftime('%I %p')} ({sleep_hours:.2f} hours).")
            except ValueError:
                st.error("Please enter valid sleep and wake times.")


def run_drowsiness_detection():
    global alert_playing, eye_closed_start_time
    stframe = st.empty()  # Create an empty container for the video frame
    try:
        vs = VideoStream(src=0).start()
        time.sleep(1.5)
    except Exception as e:
        st.error(f"Error initializing video stream: {e}")
        return
    beep_count = 0
    alert_start_time = None  # Track when the alert message started displaying

    while st.session_state.logged_in:
        frame = vs.read()
        if frame is None:
            st.error("Error: Unable to capture frame from the video stream")
            break

        frame = imutils.resize(frame, width=frame_width, height=frame_height)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape

        rects = detector(gray, 0)

        if len(rects) == 0:
            # No face detected
            cv2.putText(frame, "No Face Detected", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            if not alert_playing:
                alert_sound.play()
                alert_playing = True
            else:
                alert_sound.stop()
                alert_playing = False
                alert_start_time = None  # Reset alert start time when no face detected
        else:
            # Face detected
            for rect in rects:
                (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
                (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                if ear < EYE_AR_THRESH:
                    if eye_closed_start_time is None:
                        eye_closed_start_time = time.time()
                    elif time.time() - eye_closed_start_time >= ALERT_DURATION_THRESHOLD:
                        cv2.putText(frame, "Eyes Closed!", (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if not alert_playing:
                            alert_sound.play()
                            alert_playing = True
                            if alert_start_time is None:
                                alert_start_time = time.time()
                            beep_count += 1
                            if time.time() - alert_start_time > 5:
                                alert_sound.stop()
                                alert_playing = False
                                alert_start_time = None
                else:
                    eye_closed_start_time = None
                    if alert_playing:
                        alert_sound.stop()
                        alert_playing = False
                        alert_start_time = None

                (mStart, mEnd) = (49, 68)
                mouth = shape[mStart:mEnd]
                mouthMAR = mouth_aspect_ratio(mouth)
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
                cv2.putText(frame, "MAR: {:.2f}".format(mouthMAR), (650, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)

                if mouthMAR > MOUTH_AR_THRESH:
                    cv2.putText(frame, "Yawning!", (800, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not alert_playing:
                        alert_sound.play()
                        alert_playing = True
                        if alert_start_time is None:
                            alert_start_time = time.time()
                    beep_count += 1
                    if time.time() - alert_start_time > 5:
                        alert_sound.stop()
                        alert_playing = False
                        alert_start_time = None

                for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        current_datetime = datetime.now().strftime("%d %b %Y %I:%M:%S %p")
        time_of_day = get_time_of_day()
        cv2.putText(frame, f"Date & Time: {current_datetime}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                    2)
        cv2.putText(frame, f"Time of Day: {time_of_day}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb)

    cv2.destroyAllWindows()
    vs.stop()
def logout():
    st.session_state.logged_in = False
    st.success("Logged out successfully!")


def display_user_info():
    st.subheader("User Information")
    current_user = st.session_state.current_user

    if current_user is not None:
        st.write(f"Username: {current_user}")

        if current_user in st.session_state.sleep_data:
            sleep_history = st.session_state.sleep_data[current_user]
            st.write("Sleep Data History:")
            for entry in sleep_history:
                if isinstance(entry, tuple):
                    hours, timestamp = entry
                    st.write(f"- Hours: {hours}, Recorded at: {timestamp}")
                else:
                    st.write(f"- Hours: {entry}, Recorded at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.write("No sleep data found for this user.")
    else:
        st.write("No user selected.")

def about():
    st.subheader("About")
    st.write("This application performs drowsiness detection using computer vision techniques. It detects eye closure and yawning in real-time video streams and alerts users accordingly.")

def main():
    st.title("Drowsiness Detection System")

    # Add CSS for gradient background and sidebar color
    st.markdown(
        """
        <style>
        .stApp {
            background-image: linear-gradient(to bottom right, #d8b5ff, #1eae98);
            background-repeat: no-repeat;
            background-attachment: fixed;
            height: 100vh;
            width: 100vw;
            overflow: hidden;
        }
        [data-testid="stSidebar"] {
            background-image: linear-gradient(to bottom right, #1eae98, #d8b5ff);
            background-color: #1eae98; /* Fallback color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Home", "Create Account", "Login", "Enter Sleep Times", "Run Drowsiness Detection", "User Information", "About", "Logout"))
    if page == "Home":
        st.header("Home")
        st.write("Welcome to the Drowsiness Detection System. Use the navigation on the left to explore the application.")
    elif page == "Create Account":
        create_account()
    elif page == "Login":
        if not st.session_state.get('logged_in', False):
            login()
        else:
            st.warning("You are already logged in.")
    elif page == "Enter Sleep Times":
        if st.session_state.get('logged_in', False):
            enter_sleep_times()
        else:
            st.warning("Please log in to enter sleep times.")
    elif page == "Run Drowsiness Detection":
        if st.session_state.get('logged_in', False):
            run_drowsiness_detection()
        else:
            st.warning("Please log in to run drowsiness detection.")
    elif page == "User Information":
        if st.session_state.get('logged_in', False):
            display_user_info()
        else:
            st.warning("Please log in to view user information.")
    elif page == "About":
        about()
    elif page == "Logout":
        logout()

if __name__ == "__main__":
    main()