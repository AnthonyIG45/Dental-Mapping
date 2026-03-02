Dental X-ray Segmentation Tool
This script uses computer vision techniques to identify and visualize the gaps between teeth and the separation between the upper and lower jaw in dental X-rays.

Features
- Noise Reduction: Implements Gaussian Blurring to ensure smoother intensity analysis
- Horizontal Segmentation: Divides the image into vertical strips for local analysis
- Uses a 2nd-degree polynomial fit to create a curved baseline between jaws
- Vertical Gap Detection: Calculates intensity projections above and below the horizontal curve
- Identifies local minima to locate spaces between individual teeth.
- Visualization: Exports segmented_teeth.png with color-coded markers:
- Green Curve: Upper/Lower jaw separation
- Blue Lines: Upper teeth gaps
- Red Lines: Lower teeth gaps

Usage
- Place your dental X-ray image in the project directory.
- Update the function call at the bottom of the script with your filename:
- process_dental_xray('your_image_here.png')

Prerequisites
Ensure you have the following Python libraries installed:

         Bash:
         pip install opencv-python numpy matplotlibPrerequisites


Run the script. The processed image will be displayed and saved locally.
