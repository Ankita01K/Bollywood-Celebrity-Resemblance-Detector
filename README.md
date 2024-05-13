# **Bollywood Celebrity Resemblance Detector**

## **Overview**

**This project employs Convolutional Neural Networks (CNNs), specifically the VGG-Face architecture, to identify resemblances between uploaded images and Bollywood celebrities. By leveraging pre-trained models and advanced image processing techniques, we've created a robust system capable of suggesting the closest match from a dataset of over 8664 celebrity images.**

## **How Does it Work?**

**The core of our approach lies in utilizing VGG-Face's convolutional layers for feature extraction, discarding the final fully connected layers. By customizing this established architecture, we ensure accurate facial feature recognition. Here's a more detailed breakdown of the process:**

**1.Data Gathering : We collected over 8664 images from the 'Bollywood Celeb Localized Face' dataset on Kaggle [Here](https://www.kaggle.com/datasets/sushilyadav1998/bollywood-celeb-localized-face-dataset),  focusing solely on facial details for each celebrity.**
 

**2.Model Building : Using Spyder, we established a virtual environment and installed necessary libraries including TensorFlow, OpenCV, and Streamlit. We then utilized VGG-Face's ResNet50 model to extract facial features from the dataset.**


 + **VGG-Face : VGG-Face is a CNN architecture specifically designed for face recognition tasks. It consists of five convolutional layers and two fully connected layers, tailored for images sized (224,224,3) to extract 2048 distinct facial features. By leveraging pre- 
      trained VGG-Face models, we capitalize on extensive research and robust feature extraction capabilities.**
  

**3.Testing : We conducted thorough testing to validate the model's performance, employing image processing techniques like MTCNN for facial detection and extraction.**


 + **MTCNN (Multi-Task Cascaded Convolutional Neural Network): MTCNN is a powerful tool for face detection and alignment in images. It works by detecting faces in an image and then refining the bounding boxes to accurately align with facial features. We utilized MTCNN 
       to preprocess uploaded images, ensuring uniformity by isolating only the facial part for further analysis.**
   

**4.Recommendation Algorithm : We implemented a cosine similarity-based algorithm to suggest the most similar image from our dataset based on feature vectors extracted from uploaded images.**


**5.Website Building: Leveraging Streamlit, we developed a user-friendly interface for showcasing the project, enabling users to upload their images and receive instant resemblance suggestions.**


## **To run the project locally, follow these steps:**


+ **Clone this repository**

  > https://github.com/Ankita01K/Bollywood-Celebrity-Resemblance-Detector.git

+ **Install required dependencies using**

  > pip install -r requirements.txt. 

+ **Run the Streamlit app using**

  > streamlit run app.py
  
+ **Upload your image and discover your Bollywood celebrity resemblance**

  
## Demo

![Screenshot 2024-05-12 160832](https://github.com/Ankita01K/Bollywood-Celebrity-Resemblance-Detector/assets/123232024/03a01f46-9656-4942-8625-d227189bd038)


![Screenshot 2024-05-13 210207](https://github.com/Ankita01K/Bollywood-Celebrity-Resemblance-Detector/assets/123232024/a7dba571-e8a1-406a-b079-06988b26fbd8)

Fun Fact I Look like Taapsee Pannu :sweat_smile:

![Screenshot 2024-05-13 210646](https://github.com/Ankita01K/Bollywood-Celebrity-Resemblance-Detector/assets/123232024/ad976be8-7d8e-4a42-bc7e-9a4c9ac91998)


