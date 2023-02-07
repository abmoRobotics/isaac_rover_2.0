from zipfile import ZipFile
import glob, os

options = []
files = []
extract_zip = ""
os.chdir("./")


while True:
    for idx, file in enumerate(glob.glob("*.zip")):
        options.append(int(idx))
        files.append(file)
        print(str(idx) + ":", str(file))
    x = input("Choose terrain: ")
    if int(x) in options:
        extract_zip = files[int(x)]
        break

with ZipFile(extract_zip, 'r') as zObject:
    zObject.extractall("../")
    print("Successfully extracted the " + extract_zip[:-4] + " terrain.")

