import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_dental_xray(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found.")
        return

    # Apply reduce noise 
    smoothed = cv2.GaussianBlur(img, (15, 15), 0)
    
    # Horizontal Gap Detection
    # Calculate sum intensities along rows
    y_projection = np.sum(smoothed, axis=1)
    
    # Divide image into vertical strips
    height, width = img.shape
    num_strips = 10
    strip_width = width // num_strips
    gap_points_x = []
    gap_points_y = []

    for i in range(num_strips):
        x_start = i * strip_width
        x_end = (i + 1) * strip_width
        strip = smoothed[:, x_start:x_end]
        strip_y_proj = np.sum(strip, axis=1)
        
        # Restrict search to middle 40% of image height to avoid jaw edges
        search_min, search_max = int(height*0.3), int(height*0.7)
        gap_y = np.argmin(strip_y_proj[search_min:search_max]) + search_min
        
        gap_points_x.append(x_start + strip_width // 2)
        gap_points_y.append(gap_y)

    # Fit upper-lower gap separation
    poly_h = np.polyfit(gap_points_x, gap_points_y, 2)
    h_curve = np.poly1d(poly_h)

    # Vertical Gap Detection (Between Teeth)
    # Separate the image into upper and lower halves based on the curve
    def get_vertical_gaps(is_upper):
        x_range = np.arange(0, width, 1)
        x_projection = np.zeros(width)
        
        for x in x_range:
            mid_y = int(h_curve(x))
            # Sum pixels above or below the horizontal curve
            if is_upper:
                column_segment = smoothed[0:mid_y, x]
            else:
                column_segment = smoothed[mid_y:height, x]

            x_projection[x] = np.sum(column_segment)

        x_proj_smoothed = np.convolve(x_projection, np.ones(20)/20, mode='same')
        gap_indices = []

        for i in range(20, width - 20):
            if x_proj_smoothed[i] == min(x_proj_smoothed[i-20:i+20]):
                if len(gap_indices) == 0 or abs(i - gap_indices[-1]) > width/10:
                    gap_indices.append(i)
        return gap_indices

    upper_gaps = get_vertical_gaps(is_upper=True)
    lower_gaps = get_vertical_gaps(is_upper=False)
    output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    plot_x = np.linspace(0, width-1, width).astype(int)
    plot_y = h_curve(plot_x).astype(int)

    for i in range(len(plot_x)-1):
        cv2.line(output_img, (plot_x[i], plot_y[i]), (plot_x[i+1], plot_y[i+1]), (0, 255, 0), 2)

    for x_gap in upper_gaps:
        mid_y = int(h_curve(x_gap))
        cv2.line(output_img, (x_gap, 0), (x_gap, mid_y), (255, 0, 0), 2)
    
    for x_gap in lower_gaps:
        mid_y = int(h_curve(x_gap))
        cv2.line(output_img, (x_gap, mid_y), (x_gap, height), (0, 0, 255), 2)

    cv2.imwrite('segmented_teeth.png', output_img)
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

process_dental_xray('teeth_sample.png')