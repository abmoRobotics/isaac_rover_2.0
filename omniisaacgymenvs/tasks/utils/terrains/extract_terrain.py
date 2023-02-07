

from zipfile import ZipFile
import glob, os
import time

def extract_terrain():
    options = []
    files = []
    extract_zip = ""
    os.chdir("./")
    path = "tasks/utils/terrains/"
    print("[ERROR] No terrain defined, please choose a terrain to extract")
    while True:
        for idx, file in enumerate(glob.glob(path + "*.zip")):
            options.append(int(idx))
            files.append(file)
            formattedFileName = str(file).replace(path, "").replace(".zip", "")
            print(str(idx) + ":", formattedFileName + " terrain.")
        x = input("Choose terrain: ")
        if int(x) in options:
            extract_zip = files[int(x)]
            break

    with ZipFile(extract_zip, 'r') as zObject:
        zObject.extractall(path + "../")
        print("Successfully extracted the " + extract_zip[:-4] + " terrain.")
        time.sleep(1)

