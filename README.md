# Original project  
 •https://github.com/Utkarsh-Deshmukh/Single-Image-Dehazing-Python  

# How to run the program  
1. Open the terminal inside of the project folder  
2. Run this command "py runThis.py"  

# Changes that I made to the  
## "runThis.py" file  
 •Renamed "example.py" to "runThis.py"  
 •Created the "ClearTheOutputFolder()" function, to delete everything currently inside of the "output" folder at the beginning of the program  
 •Added a while loop, to allow the user to choose which image to dehaze in the "inputImages" folder  
 •Added "cv2.resize()", to make the image display windows smaller (they were WAY too big before)  
 •Added "time.process_time()", to record how long each algorithm takes  
 •Added "img_dehazeFaster, haze_map", to run my faster dehaze algorithm  
 •Added "cv2.imshow()", to display the new image created by "img_dehazeFaster"  
 •Added "cv2.imwrite()", to write these files inside of the "output" folder: "img_input", "img_dehazeOld", "img_dehazeFaster"  
 •Added "shutil.rmtree()", to automatically remove the cache folders after the program exits  
 •Created the "CreateOutputLogFile()" function, to write down the runtime for both "img_dehazeOld" and "img_dehazeFaster", and compare them  

## "img_dehazeFaster/__init__.py" file, to make it run faster  
 •Replaced a for loop with cv2.erode  
 •Replaced copying of arrays with storing their values  
 •Replaced manual array operations with NumPy operations  
 •Replaced multiple calculations with np.maximum.reduce()  

## "inputImages" folder  
 •Changed some of the file names to match the same naming conventions as the other files  
