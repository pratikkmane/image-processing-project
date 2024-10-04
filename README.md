# image-processing-project
This repository showcases essential computer vision and image processing techniques using Python, NumPy, OpenCV, and Matplotlib. Key features include Sobel edge detection, Gaussian smoothing, and histogram equalization, with scripts and notebooks demonstrating their applications on sample images. Contributions to enhance the project are welcome.
# introduction
This README provides step-by-step instructions for running the image processing assignment code on Google Colab and Local Server.
I have provided 5 (.ipynb) files for each question to run on google colab and 1 (.py) file to run on the local server it has the code of all 5 questions in it.
# Prerequisites
A Google account (to access Google Colab)
An internet connection
The 'Lena.png' image file (or any other image you wish to process) is in the same folder as the .ipynb or .py file.
VS Code if needed.

## Steps to Run the Code on Google Colab***

### 1. Access Google Colab
1. Open your web browser and go to [Google Colab].
2. If prompted, sign in with your Google account.
### 2. Open a Notebook
1. Click on the file and then open a notebook and select the desired code (.ipynb) file.
### 3. Upload the Image
1. Look for the following line in the code:
   ```python
   uploaded = files.upload()
   ```
2. Run this cell by clicking the play button to its left or pressing Shift+Enter.
3. A "Choose Files" button will appear below the cell.
4. Click this button and select your 'Lena.png' file (or whichever image you're using).
### 4. Run the Code
1. After uploading the image, the code will automatically continue executing.
2. If it doesn't, click the play button again or press Shift+Enter.

## Steps to Run the Code on Local Server***

### 1. Access Command Prompt / VS Code / Terminal
1. Make sure Python is installed on your system or go to google and download & install python.
2. Use python --version command in Terminal to check if it's properly installed on your system.
3. Open any of the above in the folder where the provided .py file and image are located.
4. Just run the following command: python filename.py and press Enter.
### After executing the file each output window will appear one after another. You have to close the first output window then the second one will appear since it is a big project. The runtime will vary depending on the specific machine.
## Troubleshooting
- If you encounter an error about missing libraries, add the following at the beginning of your code and run it:
  ```python
  !pip install numpy matplotlib pillow
  ```

## Conclusion
By following these steps, you should be able to successfully run this image processing project code on Google Colab or Local Server.
