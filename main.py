import ComputerVision.Image_Processing as file1
import ComputerVision.Video_Processing as file2
import sys


if sys.argv[1] == "file1":
    if __name__ == '__main__':
        file1.main()

elif sys.argv[1] == "file2":
    if __name__ == '__main__':
        file2.main()
else:
    print("Invalid argument")
