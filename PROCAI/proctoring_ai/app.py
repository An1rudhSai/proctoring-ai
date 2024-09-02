from flask import Flask, render_template, request, jsonify
import subprocess
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
import threading
import time
import json
import os
import requests

app = Flask(__name__)

# Constants and Configuration
EMAIL_SENDER = os.getenv('EMAIL_SENDER', 'your_email@gmail.com')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', 'your_password')  # Ideally, use an environment variable
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
FINAL_SCRIPT = os.getenv('FINAL_SCRIPT', 'path/to/your/final.py')  # Adjusted for environment
REPORT_FILENAME = os.getenv('REPORT_FILENAME', 'report.json')  # Adjusted for environment

# Global variable to hold the process and timing
process = None
start_time = None
report_email_data = None

def send_email(subject, body, recipient_email):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = recipient_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def write_report(report_data):
    # Ensure the directory exists before writing to the file
    os.makedirs(os.path.dirname(REPORT_FILENAME), exist_ok=True)
    
    # Write the report data to the file
    with open(REPORT_FILENAME, 'w') as file:
        json.dump(report_data, file, indent=4)

def monitor_process():
    global process, start_time, report_email_data

    while process.poll() is None:
        time.sleep(1)  # Check every second

    # Process has finished
    end_time = datetime.now()
    duration = end_time - start_time

    # Try to read the report.json file if available
    try:
        with open(REPORT_FILENAME, 'r') as file:
            report_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        report_data = {
            'name': report_email_data['name'],
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration': str(duration),
            'total_run_time': 'Not available',
            'head_tilt_time': 'Not available',
            'mouth_opening_time': 'Not available',
            'phone_detected': 'Not available',
            'multiple_people_detected': 'Not available',
        }

    # Write the report to a file (ensure the latest data is saved)
    write_report(report_data)

    # Send the email
    if report_email_data and 'name' in report_email_data and 'email' in report_email_data:
        full_output = (f"Report for {report_email_data['name']}\n\n"
                       f"Started at: {start_time}\nEnded at: {end_time}\nDuration: {duration}\n\n"
                       f"Total run time: {report_data.get('total_run_time', 'Not available')}\n"
                       f"Head tilt time: {report_data.get('head_tilt_time', 'Not available')}\n"
                       f"Mouth opening time: {report_data.get('mouth_opening_time', 'Not available')}\n"
                       f"Phone detected: {report_data.get('phone_detected', 'Not available')}\n"
                       f"Multiple people detected: {report_data.get('multiple_people_detected', 'Not available')}\n")
        send_email("Proctoring Report", full_output, report_email_data['email'])
        print("Email Sent!")  # Print the message for debugging

    # Notify the front-end that the process has ended
    with app.app_context():
        public_url = os.getenv('RENDER_EXTERNAL_URL', 'http://127.0.0.1:5000')
        requests.post(f'{public_url}/notify_end')

@app.route("/", methods=["GET", "POST"])
def index():
    global process, start_time, report_email_data

    if request.method == "POST":
        action = request.form.get("action")

        if action == "start":
            name = request.form.get("name")
            email = request.form.get("email")

            if name and email:
                report_email_data = {'name': name, 'email': email}

                # Start the proctoring script
                start_time = datetime.now()
                process = subprocess.Popen(['python3', FINAL_SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Start the monitor thread
                threading.Thread(target=monitor_process).start()

                return jsonify({'success': True, 'message': 'Proctoring started successfully. This model runs for a default of 1 minute.'})
            else:
                return jsonify({'success': False, 'message': 'Name and email are required.'})

        elif action == "quit":
            if process and process.poll() is None:
                process.terminate()  # Terminate the running process

            return jsonify({'success': True, 'message': 'Proctoring stopped and report sent.'})

    return render_template("index.html")

@app.route("/notify_end", methods=["POST"])
def notify_end():
    return jsonify({'success': True, 'message': 'Proctoring has ended'})

if __name__ == "__main__":
    # Use the environment variable PORT for Render deployment or default to 5000 for local development
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
