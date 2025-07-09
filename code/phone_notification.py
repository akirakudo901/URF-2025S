# Taken from the following GitHub Gist:
# https://gist.github.com/cedrickchee/420061a44553973473476005b3a16cbc#file-send_notification-py

from datetime import datetime
import os
import requests
import yaml

def send_notification(msg, auth_info_path : str="/root/URF-2025S-AkiraKudo/URF-2025S/PRIVATE_notification_info.yaml"):
    """
        Send message to mobile using Pushover notifications.
        Calls Pushover API to do that.
        Pushover API docs: https://pushover.net/api
    """
    if not os.path.exists(auth_info_path):
        raise Exception(f"auth_info_path does not exist, please point to the right file: \n {auth_info_path}")
    
    with open(auth_info_path, 'r') as f:
        auth_info = yaml.safe_load(f)

    url = "https://api.pushover.net/1/messages.json"
    data = {
        "user"  : auth_info["user"],
        "token" : auth_info["token"],
        "sound" : "magic"
    }
    data["message"] = msg
    data['message'] = data['message'] + "\n" + str(datetime.now())

    r = requests.post(url = url, data = data)

if __name__ == "__main__":
    # send_notification("Testing message sending from the Linux machine.")
    pass