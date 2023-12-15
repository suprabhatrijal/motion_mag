import numpy as np
import cv2
import pandas as pd
import gc
import os
import matplotlib.pyplot as plt
# import easygui


# video_path = easygui.enterbox("Input the full path of the video you want to track!")
video_name = 'Camera_A'

video_path = f'./{video_name}.mp4'



def write_to_csv_content(content,p_data,p_path):
    """Save a Pandas dataframe to a .csv file."""
    dataframe = pd.DataFrame(p_data)
    if content.lower() == 'title':
        if (os.path.exists(p_path)):
            os.remove(p_path)
    elif content.lower() == 'coordinates':
        pass
    else:
        os.error('Unexpected content')    
    dataframe.to_csv(p_path, mode='a',header=False,index=False,sep=',')
    del dataframe
    gc.collect()
    

cap = cv2.VideoCapture(str(video_path))

ref, firstframe = cap.read()
height, width, channels = firstframe.shape
old_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
global cache
cache = firstframe.copy()

first_point_size = 10
font = cv2.FONT_HERSHEY_SIMPLEX

all_points = list() #list to hold all selected points of interest
point_num = 0 # number of points of interest

# flag to determine if the user is panning or just moving the mouse
# triggered by rclick
Panning = False

# defining default values for the panning and zooming movements

offsetX_Value = 0
offsetY_Value = 0
startPanX = 0
startPanY = 0
zoomScaleX = 1
zoomScaleY = 1
max_points = 0

# a function to reset said values
def defaultPanZoomValues():
    global offsetX_Value, offsetY_Value, startPanX, startPanY, zoomScaleX,zoomScaleY
    offsetX_Value = 0
    offsetY_Value = 0
    startPanX = 0
    startPanY = 0
    zoomScaleX = 1
    zoomScaleY = 1


show_feature_paths = True
save_paths = False
save_no_paths = False
csv_path_all_points = './motion_coordinates_all_points.csv'

configure = "L" # Lucas-Kanade configuration

# functions to translate from screen coordinates (physical window) to world coordinates (actual video) and vice versa
# origin point is assumed to be top left corner (0,0)
def worldToScreen(worldX, worldY, offsetX, offsetY):
    screenX = int((worldX + offsetX)*zoomScaleX)
    screenY = int((worldY + offsetY)*zoomScaleY)
    return screenX, screenY
def screenToWorld(screenX, screenY, offsetX, offsetY):
    worldX = int((screenX/zoomScaleX - offsetX))
    worldY = int((screenY/zoomScaleY - offsetY))
    return worldX, worldY

if (save_paths or save_no_paths):
        base = os.path.splitext(os.path.basename(video_path))[0]
        vid_sn_paths = base + '_analysis_paths_shown.avi'
        vid_sn_no_paths = base + '_analysis_paths_not_shown.avi'

#choose how you want to specify points
# is_points_give = easygui.buttonbox("Point specification method (pixel or manual):", "Specification method", ["pixel", "manual"])
is_points_give = 'manual'
print(is_points_give)

if is_points_give == 'pixel':
    # Number of points(s)
    while True:
        isNum = True
        num_points = 300
        try:
            num_points = int(num_points)
            break
        except ValueError:
            print('"{}" is not a valid number, please try again.'.format(num_points))
            isNum = False
            continue
    # Point location(s)
    while point_num < num_points:
        point_coords = input("The x,y coordinates of Point " + str(point_num+1) + " (e.g., 682,184):\n")
        get_point = np.array([x.strip() for x in point_coords.split(',')])           
        try:
            val1, val2 = int(get_point[0]), int(get_point[1])
            all_points.append([val1, val2])
            point_num += 1
        except ValueError:
            print('"{}" is not a valid integer pair, please try again.'.format(point_coords))              
            pass
        except IndexError:
            print('"{}" does not specify two points, please try again.'.format(point_coords))
            pass
    max_points = num_points
# Choose key points manually
elif is_points_give == 'manual':
    
    cv2.namedWindow('frame test',cv2.WINDOW_AUTOSIZE) #cv2.WINDOW_AUTOSIZE

    def instructions():
        cv2.putText(firstframe, 'Press Esc or Q to Exit', (100, 40), font, 0.75, (255, 255, 255), 2)
        cv2.putText(firstframe, 'Press R to reset panning and zoom', (100, 80), font, 0.75, (255, 255, 255), 2)
        cv2.putText(firstframe, 'Right Click to Pan', (100, 120), font, 0.75, (255, 255, 255), 2)
        cv2.putText(firstframe, 'Scroll to Zoom', (100, 160), font, 0.75, (255, 255, 255), 2)
        cv2.putText(firstframe, 'Press Z to remove last point', (100, 200), font, 0.75, (255, 255, 255), 2)
        cv2.putText(firstframe, 'Press C to clear all points', (100, 240), font, 0.75, (255, 255, 255), 2)
    instructions()
   
    def mouse_events(event, x, y, flags, param):
        global Panning, offsetX_Value, offsetY_Value, startPanX, startPanY, zoomScaleX,zoomScaleY
        #Placing points
        if event == cv2.EVENT_LBUTTONDOWN:
            circle_x, circle_y = screenToWorld(x,y,offsetX_Value,offsetY_Value) #converting points in screenspace to world space
            #cv2.circle(frame, (circle_x, circle_y), first_point_size, (0, 0, 0), -1)
            cv2.circle(firstframe, (circle_x, circle_y), first_point_size-6, (255, 255, 255), -1)               
            cv2.putText(firstframe, f"({circle_x}, {circle_y})", (int(circle_x-25), int(circle_y-25)), font, 1.25, (255, 255, 255), 3)
            all_points.append([circle_x, circle_y]) #TODO have it so that these points can be reajusted afterwards for math 

        #Panning the image
        if event == cv2.EVENT_RBUTTONDOWN:
            Panning = True
            startPanX,startPanY = x,y #Saving current mouse position as a reference origin poinr
        elif event ==cv2.EVENT_MOUSEMOVE and Panning is True:
            #print(f"moving {x}, {y}")
            
            # offset is based on the difference between the current coords and the reference
            offsetX_Value += (x-startPanX)
            offsetY_Value += (y-startPanY) 
            # reset reference
            startPanX = x
            startPanY = y
            
        elif event == cv2.EVENT_RBUTTONUP:
            Panning = False #reset spanning flag

        #Zooming Image
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 1:
                zoomScaleX *= 1.01
                zoomScaleY *= 1.01
            elif flags< 1:
                zoomScaleX *= 0.99
                zoomScaleY *= 0.99
            
# https://youtu.be/ZQ8qtAizis4 
    while True:
        cv2.setMouseCallback('frame test', mouse_events)    
    
        T = np.float32([[1, 0, offsetX_Value], [0, 1, offsetY_Value]]) 
        frame_translate = cv2.warpAffine(firstframe, T, (width, height)) #https://www.geeksforgeeks.org/image-translation-using-opencv-python/

        widthZoom = int((width)*zoomScaleX)
        heightZoom = int((height)*zoomScaleY)

        frame_translate_zoom = cv2.resize(frame_translate, (widthZoom,heightZoom))
        cv2.imshow('frame test', frame_translate_zoom)
        
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        if key == ord('q'):
            break
        if key == ord('r'): #reset offsets and zoom scales
            defaultPanZoomValues()
            print("reset")
        if key == ord("z"): # undo button
            if len(all_points) > 0:
                p = all_points.pop() # remove last point added to list - like a stack
                firstframe = cache
                cache = firstframe.copy()
                instructions() # reprint instructions
                for x,y in all_points: 
                    cv2.circle(firstframe, (x, y), first_point_size-6, (255, 255, 255), -1)
                    cv2.putText(firstframe, f"({x}, {y})", (int(x-25), int(y-25)), font, 1.25, (255, 255, 255), 3)
        if key == ord("c"): # clear all points on screen and from memory
            firstframe = cache
            cache = firstframe.copy()
            instructions()
            all_points = list()
    max_points = len(all_points)

    cv2.destroyAllWindows()

ref_pts = []
ref_pt_inp = is_points_give = "n"

if ref_pt_inp == 'y':
    ref, ref_frame = cap.read()
    defaultPanZoomValues()
    cv2.namedWindow('refframe')
    cv2.putText(ref_frame, 'Press Escape after selecting desired point(s).', (100, 40), font, 0.75, (255, 255, 255), 3)
    
    

    def mouse_events2(event, x, y, flags, param):
        global Panning, offsetX_Value, offsetY_Value, startPanX, startPanY, zoomScaleX,zoomScaleY
        
        if event == cv2.EVENT_LBUTTONDOWN:
            circle_x, circle_y = screenToWorld(x,y,offsetX_Value,offsetY_Value) #converting points in screenspace to world space
            #TODO have it so that these points can be reajusted afterwards for math 

            cv2.circle(ref_frame, (circle_x, circle_y), first_point_size, (0, 0, 0), -1)
            cv2.circle(ref_frame, (circle_x, circle_y), first_point_size-4, (0, 0, 255), -1)          
            cv2.putText(ref_frame, f"({circle_x}, {circle_y})", (int(circle_x-25), int(circle_y-25)), font, 1.25, (255, 255, 255), 3)   
            ref_pts.append([circle_x, circle_y])

        if event == cv2.EVENT_RBUTTONDOWN:
            Panning = True
            startPanX,startPanY = x,y #Saving current mouse position as a reference origin poinr
        elif event ==cv2.EVENT_MOUSEMOVE and Panning is True:
            #print(f"moving {x}, {y}")
            
            # offset is based on the difference between the current coords and the reference
            offsetX_Value += (x-startPanX)
            offsetY_Value += (y-startPanY) 
            # reset reference
            startPanX = x
            startPanY = y
            
        elif event == cv2.EVENT_RBUTTONUP:
            Panning = False #reset spanning flag

        #Zooming Image
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 1:
                zoomScaleX *= 1.01
                zoomScaleY *= 1.01
            elif flags< 1:
                zoomScaleX *= 0.99
                zoomScaleY *= 0.99
    while True:

        cv2.setMouseCallback('refframe', mouse_events2)
        T = np.float32([[1, 0, offsetX_Value], [0, 1, offsetY_Value]]) 
        frame_translate = cv2.warpAffine(ref_frame, T, (width, height)) #https://www.geeksforgeeks.org/image-translation-using-opencv-python/

        widthZoom = int((width)*zoomScaleX)
        heightZoom = int((height)*zoomScaleY)

        frame_translate_zoom = cv2.resize(frame_translate, (widthZoom,heightZoom))
        cv2.imshow('refframe', frame_translate_zoom)            
        k = cv2.waitKey(20) & 0xFF
        if len(ref_pts) == 2:
            break

    cv2.destroyAllWindows()
    # Reference length
    ref_length = ""
    while True:
        ref_length = 10
        try:
            # ref_length = int(ref_length)
            ref_length = float(ref_length)
            break
        except ValueError:
            print('"{}" is not a valid number, please try again.'.format(ref_length))
            continue    
    dx = abs(ref_pts[1][0] - ref_pts[0][0])
    dy = abs(ref_pts[1][1] - ref_pts[0][1])
    dres = (dx**2 + dy**2)**(1/2)
    length_per_pixel = ref_length / dres
else:
    length_per_pixel = 0

# https://www.wikiwand.com/en/Kanade%E2%80%93Lucas%E2%80%93Tomasi_feature_tracker
# https://www.geeksforgeeks.org/python-opencv-optical-flow-with-lucas-kanade-method/#
# https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
# https://docs.opencv.org/4.x/db/d7f/tutorial_js_lucas_kanade.html  
# https://learnopencv.com/optical-flow-in-opencv/


cv2.namedWindow('finalframe',cv2.WINDOW_AUTOSIZE) #cv2.WINDOW_AUTOSIZE
ref, frame = cap.read()
frame_i = 1
frames_num = cap.get(7)


# Define Lucas-Kanade parameters
if str(configure).upper()=="H":
    current_params = dict(
        winSize=(512, 64),
        maxLevel=8,
        flags=1,
        minEigThreshold=0.081,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50000, 0.00001)
        )
elif str(configure).upper() == "M":
    current_params = dict(
        winSize=(200, 200),
        maxLevel=6,
        flags=1,
        minEigThreshold=0.075,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 0.006)
        )
elif str(configure).upper() == "L":
    current_params = dict(
        winSize  = (15,15),
        maxLevel = 1,
        flags = 1,
        minEigThreshold=0.00025,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
# TODO: Just repeat the list a # of times equal to the number of points
lk_params_point = []
for idx in range(max_points):
    lk_params_point.append(current_params)

# Initialize (x, y) positions for points of interest based on region of interest selected
p0_point = []

for idx in range(len(all_points)):
    p0_point.append(np.array(all_points[idx],dtype=np.float32).reshape(-1,1,2))

# Initialize column headers [frame # and (x, y) positions] in .csv file for all points of interest
title = ['frame id']
for idx in range(len(all_points)):
    title.append('x_' + str(idx))
    title.append('y_' + str(idx))
title_ = np.array(title)
write_to_csv_content('title', title_.reshape(1, title_.shape[0]), csv_path_all_points)

# Define the codec and create VideoWriter object
if (save_paths or save_no_paths):
    # fourcc = int(cap.get(6)) # fourcc = cv2.VideoWriter_fourcc(*'DIVX')      
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')                       
    output_video_paths = cv2.VideoWriter(vid_sn_paths, fourcc, cap.get(5), (int(cap.get(3)), int(cap.get(4)))) # (width, height)
    output_video_no_paths = cv2.VideoWriter(vid_sn_no_paths, fourcc, cap.get(5), (int(cap.get(3)), int(cap.get(4)))) # (width, height)


mask = np.zeros_like(frame)
color = np.random.randint(0,255,(max_points,3))

defaultPanZoomValues()

def mouse_events3(event, x, y, flags, param):
    global Panning, offsetX_Value, offsetY_Value, startPanX, startPanY, zoomScaleX,zoomScaleY

    if event == cv2.EVENT_RBUTTONDOWN:
        Panning = True
        startPanX,startPanY = x,y #Saving current mouse position as a reference origin poinr
    elif event ==cv2.EVENT_MOUSEMOVE and Panning is True:
        #print(f"moving {x}, {y}")
        
        # offset is based on the difference between the current coords and the reference
        offsetX_Value += (x-startPanX)
        offsetY_Value += (y-startPanY) 
        # reset reference
        startPanX = x
        startPanY = y
    elif event == cv2.EVENT_RBUTTONUP:
        Panning = False #reset spanning flag

    #Zooming Image
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 1:
            zoomScaleX *= 1.01
            zoomScaleY *= 1.01
        elif flags< 1:
            zoomScaleX *= 0.99
            zoomScaleY *= 0.99

# Get the frame rate
fps = cap.get(cv2.CAP_PROP_FPS)
# Get the total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(fps, total_frames)

x = np.array([])
time = np.array([])
y = np.array([])
# time = np.array([])


# points = [[] for _ in range(len(all_points))]


while(True):
    # Capture frame-by-frame
    if frame_i%100 == 0:
        print("The current frame is ", frame_i, "")
    ref, frame = cap.read()
    if not ref:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    coordinates_point = []
    coordinates_point.append('frame_' + str(frame_i))
    
    for idx in range(len(all_points)):       
        # x = np.array([])
        # time = np.array([])
        # y = np.array([])        
        p1_point, st_point, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0_point[idx], None, **lk_params_point[idx])
        
        good_new_point = p1_point[st_point == 1]
        good_old_point = p0_point[idx][st_point == 1]

        st_point_element = np.array(np.where(st_point.reshape(st_point.shape[0],) == 0))[0]
        lost_st_point_element = len(st_point_element)
        if lost_st_point_element > 0:
            p1_point[st_point == 0] = p0_point[idx][st_point == 0]
            good_old_point = p0_point[idx][0]
            good_new_point = p1_point[0]
        else:
            good_old_point = p0_point[idx][st_point == 1]
            good_new_point = p1_point[st_point == 1]
        
        for ref, (new_point, old_point) in enumerate(zip(good_new_point, good_old_point)):
            a_point, b_point = new_point.ravel()
            c_point, d_point = old_point.ravel()
            a_point, b_point, c_point, d_point = int(a_point), int(b_point), int(c_point), int(d_point)

            if (show_feature_paths or save_paths):
                mask = cv2.line(mask, (a_point, b_point), (c_point, d_point), color[idx].tolist(), 6)

            x = np.append(x, a_point)
            y = np.append(y, b_point)
            # points[idx].append(x)
            # print(a_point,idx, frame_i)
            if len(time) == 0:
                time = np.append(time, 0)

            else:
                time = np.append(time, time[-1]+(1/fps))

            frame = cv2.circle(frame, (a_point, b_point), first_point_size, (0, 0, 0), -1)
            frame = cv2.circle(frame, (a_point, b_point), first_point_size-4, (255, 255, 255), -1)
            cv2.putText(frame, str(idx), (int(a_point-25), int(b_point-25)), font, 1.25, (255, 255, 255), 3)
            
            # coordinates_point.extend([i*length_per_pixel for i in [a_point, b_point]])
            # print(points)
        # points[idx].append(x)
            
        p0_point[idx] = good_new_point.reshape(-1, 1, 2)
    # print(points)
        
    # coordinates_point_ = np.array(coordinates_point)
    # print(coordinates_point)
    # write_to_csv_content('coordinates', coordinates_point_.reshape(1, coordinates_point_.shape[0]), csv_path_all_points)

    cv2.setMouseCallback('finalframe', mouse_events3)
    #print(offsetX_Value,offsetY_Value)
    cv2.putText(frame, 'EL05E42', (725, 235), font, 1.75, (255, 255, 255), 3)
    img = cv2.add(frame, mask)

    T = np.float32([[1, 0, offsetX_Value], [0, 1, offsetY_Value]]) 
    frame_translate = cv2.warpAffine(img, T, (width, height)) #https://www.geeksforgeeks.org/image-translation-using-opencv-python/

    widthZoom = int((width)*zoomScaleX)
    heightZoom = int((height)*zoomScaleY)

    frame_translate_zoom = cv2.resize(frame_translate, (widthZoom,heightZoom))
    
    # cv2.imshow('finalframe', frame_translate_zoom)    

    cv2.imshow('finalframe', img)
    
    # Terminate the video early, if desired
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break
    if key == ord('q'):

        break
    if key == ord('r'): #reset offsets and zoom scales
        defaultPanZoomValues()
        print("reset")
    # Write the video frame to VideoWriter object
    
    # if save_paths:
    #     output_video_paths.write(frame_translate_zoom)
    # if save_no_paths:
    #     output_video_no_paths.write(frame)

    # Update the previous frame and points
    old_gray = frame_gray.copy()
    frame_i = frame_i + 1

cv2.destroyAllWindows()
cap.release()

df = pd.DataFrame({"x_movement/pixel": x, "y_movement/pixel": height-y, "time/seconds": time})

df.to_csv(f"./pixel_tracking_results/{video_name}/{video_name}_data.csv")

# print(x.shape)
# print(time.shape)
# Add labels
plt.figure()
plt.xlabel("time/sec")
plt.ylabel("postion/pixel")
plt.plot(time,x)
plt.savefig(f"./pixel_tracking_results/{video_name}/{video_name}_x.png")
plt.figure()
plt.xlabel("time/sec")
plt.ylabel("postion/pixel")
plt.plot(time,height-y)
plt.savefig(f"./pixel_tracking_results/{video_name}/{video_name}_y.png")
