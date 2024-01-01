import cv2
import os
import shutil
import time
import img_dehazeFaster
import img_dehazeOld

def AfterInputIsSuccessful(inputFileName):
    print("It worked! Now please wait for it to finish")
    img_input = cv2.imread("inputImages/" + inputFileName)
    img_input = cv2.resize(img_input, (600, 450))

    img_dehaze_faster, runtime_faster = DehazeAndRecordRuntime(img_dehazeFaster, img_input)
    time.sleep(1)
    img_dehaze_old, runtime_old = DehazeAndRecordRuntime(img_dehazeOld, img_input)
    
    cv2.imshow("Original image", img_input)
    cv2.imshow("FASTER algorithm, dehazed photo", img_dehaze_faster)
    cv2.imshow("OLD algorithm, dehazed photo", img_dehaze_old)
    
    cv2.imwrite("output/img_input.png", img_input)
    cv2.imwrite("output/img_dehazeFaster.png", img_dehaze_faster)
    cv2.imwrite("output/img_dehazeOld.png", img_dehaze_old)
    
    shutil.rmtree("img_dehazeFaster/__pycache__")
    shutil.rmtree("img_dehazeOld/__pycache__")
    print("Everything has been saved inside of the \"output\" folder\n")
    
    CreateOutputLogFile(runtime_faster, runtime_old)
    cv2.waitKey(0)

def ClearTheOutputFolder():
    for item in os.listdir("output"):
        item_path = os.path.join("output", item)
        
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def DehazeAndRecordRuntime(img_dehaze, img_input):
    start = time.process_time()
    img_dehazed, haze_map = img_dehaze.remove_haze(img_input, showHazeTransmissionMap=False)
    end = time.process_time()
    runtime = end - start
    return img_dehazed, runtime

def CreateOutputLogFile(runtime_faster, runtime_old):
    runtime_faster = round(runtime_faster, 5)
    runtime_old = round(runtime_old, 5)
    
    subtractionDiff = str(round(runtime_old - runtime_faster, 5))
    divisionDiff = str(round(runtime_faster / runtime_old, 5))
    runtime_faster = str(runtime_faster)
    runtime_old = str(runtime_old)

    f = open("output/log.txt", "x")
    f.write("img_dehazeFaster, runtime ---> " + runtime_faster + " seconds\n")
    f.write("img_dehazeOld, runtime ------> " + runtime_old    + " seconds\n")
    f.write(runtime_old    + " - " + runtime_faster + " = " + subtractionDiff + ", img_dehazeFaster is " + subtractionDiff + " seconds faster than img_dehazeOld\n")
    f.write(runtime_faster + " / " + runtime_old    + " = " + divisionDiff    + ", img_dehazeFaster only takes " + divisionDiff + "× as long as img_dehazeOld")
    f.close()
    
    f = open("output/log.txt", "r")
    print(f.read())
    f.close()
    
if __name__ == "__main__":
    print("\n", end="")
    ClearTheOutputFolder()
    
    allInputFileNames = {}
    files = [f for f in os.listdir("inputImages") if os.path.isfile(os.path.join("inputImages", f))]
    for i,el in enumerate(files):
        allInputFileNames[str(i)] = el

    while(True):
        print("Which image file would you like to dehaze?")
        for i,el in enumerate(files):
            if(i <= 9):
                print(" " + str(i) + ".  " + el)
            else:
                print(" " + str(i) + ". "  + el)
                
        print("\n", end="")
        numInput = input("Enter the # here = ")
        
        if(numInput in allInputFileNames.keys()):
            inputFileName = allInputFileNames[numInput]
            AfterInputIsSuccessful(inputFileName)       
            break
        else:
            print("\"" + str(numInput) + "\" is not one of the number options")
            print("_________________________\n")