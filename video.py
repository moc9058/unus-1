import cv2

is_working = True
dev_port = 0
working_ports = []
available_ports = []
while is_working:
    camera = cv2.VideoCapture(dev_port)
    if not camera.isOpened():
        is_working = False
        print(f"Port {dev_port} is not working")
    else:
        print(f"Port {dev_port} is working")
    dev_port+=1