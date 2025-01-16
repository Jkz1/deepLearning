from base64 import encodebytes
import cv2
import io
import numpy as np
import os
import tempfile
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
# Membuat direktori untuk menyimpan hasil
os.makedirs("cropped_images", exist_ok=True)
os.makedirs("processed_images", exist_ok=True)
label_word = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def save_final_images(images_list, folder_path):
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Iterate over the images in the list
    for i, image in enumerate(images_list):
        try:
            # Scale the image from [0, 1] range to [0, 255]
            image = (image * 255).astype('uint8')  # Convert to uint8 type with proper range
            
            # Check if the image is 2D (grayscale)
            if len(image.shape) == 2:  # 2D grayscale image
                img = Image.fromarray(image, mode='L')
            elif len(image.shape) > 2:  # 3D grayscale (stack of 2D images or volumetric)
                # If it's 3D (like a stack of 2D images or a volume), process each slice
                for j in range(image.shape[0]):
                    img_slice = Image.fromarray(image[j], mode='L')
                    img_slice.save(os.path.join(folder_path, f'img_{i}_{j}.png'))
                    print(f"Image {i}_{j} saved successfully.")
                continue
            else:
                raise ValueError("Unsupported image shape: {}".format(image.shape))

            # Save the single grayscale image
            img.save(os.path.join(folder_path, f'img_{i}.png'))
        
        except Exception as e:
            print(f"Error saving image {i}: {e}")

def delete_files_in_directory(path):
    # Check if the directory exists
    if os.path.exists(path) and os.path.isdir(path):
        # Iterate over the files in the directory
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                # Check if it's a file (not a subdirectory)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

def load_image(path):
    """Membaca gambar dari path yang diberikan."""
    img = cv2.imread(path)
    cv2.imwrite('processed_images/original_image.jpg', img)
    return img

def convert_to_grayscale(img):
    """Mengonversi gambar menjadi grayscale dan menyimpannya."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('processed_images/grayscale.jpg', gray)
    return gray

def apply_threshold(gray):
    """Mengaburkan dan menerapkan adaptive threshold untuk menghasilkan gambar biner."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite('processed_images/blurred.jpg', blurred)
    
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 45, 15
    )
    cv2.imwrite('processed_images/threshold.jpg', thresh)
    return thresh

def find_contours(thresh):
    """Mencari kontur dalam gambar biner."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_bounding_boxes(img, contours):
    """Menggambar bounding box di sekitar kontur dan menyimpan hasilnya."""
    img_with_boxes = img.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imwrite('processed_images/bounding_boxes.jpg', img_with_boxes)
    return img_with_boxes

def save_cropped_images(img, contours):
    """Save cropped images from contours and return their paths."""
    cropped_image_paths = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if(w <= 30 or h <= 30):
                continue
            else:
                cropped = img[y:y+h, x:x+w]  # Crop the image to the bounding box of the contour
                crop_path = os.path.join(temp_dir, f"crop_{i}.png")
                cv2.imwrite(crop_path, cropped)
                cv2.imwrite(f"cropped_images/crop_{i}.png", cropped)
                cropped_image_paths.append(crop_path)
                
def do_detection():
    images = []
    # Iterate over the images in the folder
    for filename in os.listdir('cropped_images'):
        file_path = os.path.join('cropped_images', filename)
        
        # Check if it's an image file (optional check for image extensions)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Read the image
            img = cv2.imread(file_path)
            
            # Append the image to the list
            if img is not None:  # Ensure the image is loaded correctly
                images.append(img)

    model = tf.keras.models.load_model('TextRecogDNN.h5')
    processed_images = []
    
    for img in images:

        # Resize gambar menjadi 100 x 100
        resized_img = cv2.resize(img, (100, 100))
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        # Tambahkan ke list hasil
        processed_images.append(gray)
    
    processed_images = np.array(processed_images)
    processed_images = processed_images.astype('float32') / 255.0
    # processed_images[0].shape
    
    rawPredict = model.predict(processed_images)
    predictLabel = np.argmax(rawPredict, axis=1)
    delete_files_in_directory('cropped_images')
    delete_files_in_directory('processed_images')
    tmp = []
    
    for i in range (len(predictLabel)):
        tmp.append(label_word[predictLabel[i]])

    predictLabel = tmp
    save_final_images(processed_images, 'final_images')
    encode = []
    for i in range (len(os.listdir('final_images'))):
        pil_img = Image.open('final_images/img_'+str(i)+".png", mode='r') # reads the PIL image
        byte_arr = io.BytesIO()
        pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
        encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
        encode.append(encoded_img)
    
    response = {
        "image" : encode,
        "label" : predictLabel
    }
    return response

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    else:
        print(f"Directory '{path}' already exists.")

def execute():
    create_directory('processed_images')
    create_directory('final_images')
    create_directory('cropped_images')
    # Menggunakan fungsi-fungsi di atas
    img = load_image('image.png')
    gray = convert_to_grayscale(img)
    thresh = apply_threshold(gray)
    contours = find_contours(thresh)
    save_cropped_images(img, contours)
    response = do_detection()
    delete_files_in_directory('final_images')
    return response