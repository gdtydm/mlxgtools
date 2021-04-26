import os 

def mount_google_drive(mount_path='/content/gdrive'):
    from google.colab import drive
    drive.mount(mount_path)
    



