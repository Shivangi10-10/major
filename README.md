Smart Traffic Management System with Federated Learning
This project aims to improve traffic management by dynamically adjusting traffic light durations using federated learning and object detection. Vehicles at an intersection are detected in real-time, and green light duration is adjusted based on traffic load, prioritizing larger vehicles to optimize traffic flow.

Project Overview
Federated Learning: Utilizes a federated learning framework to train a model across multiple traffic light locations, ensuring data privacy by only sharing model weights, not raw data.
YOLOv8 for Object Detection: YOLOv8 is used to detect vehicles and pedestrians in video frames.
Traffic Light Control: Green light duration is dynamically adjusted based on vehicle type and count, giving priority to larger vehicles (like buses and trucks) to manage congestion effectively.
Requirements
Python 3.x
Libraries: torch, torchvision, opencv-python-headless, tqdm, scikit-learn, pandas, seaborn, ultralytics
To install dependencies:

bash
Copy code
pip install torch torchvision torchaudio opencv-python-headless tqdm scikit-learn pandas seaborn ultralytics
Project Structure
extract_frames: Function to extract frames from uploaded video files for object detection.
detect_vehicles: Function to detect vehicles in frames using YOLOv8.
TrafficDataset: Custom dataset class for handling vehicle images and labels.
train_local_model: Function to train individual client models in federated learning.
federated_avg: Implements the FedAvg algorithm to average model weights across clients.
TrafficLightController: Class that calculates green light duration based on detected vehicle types and counts.
How it Works
Data Collection: Video footage of traffic is uploaded, and frames are extracted.
Object Detection: YOLOv8 model detects vehicles, classifying them into categories such as cars, buses, and trucks.
Federated Learning:
Each traffic light acts as a federated client, training a local model on its own data.
Models from each client are aggregated to form a global model using the FedAvg algorithm, allowing the system to learn without centralizing data.
Traffic Light Control:
The system calculates green light duration based on the weighted count of detected vehicles.
Federated Learning Algorithm
The FedAvg algorithm was used for model aggregation. In FedAvg, each client trains a local model, and the weights are averaged to create a global model. This ensures privacy and reduces the need for centralized data.

Evaluation
Model accuracy is calculated by checking the correct classifications of vehicle types in the test dataset. The final accuracy provides an assessment of the model’s ability to classify vehicles correctly.

Example Usage
python
Copy code
# Initialize and use the traffic light controller
controller = TrafficLightController(global_model)
sample_frame = video_frames[0]  # Sample frame
green_duration = controller.control_traffic_light(sample_frame)
print(f"Recommended green light duration: {green_duration} seconds")
Visualization
The visualize_results function allows users to visualize detected vehicles and recommended green light duration on a sample frame, highlighting the system’s functionality.

Future Improvements
Enhance model by integrating more complex CNN architectures.
Increase dataset size and diversity.
Implement real-time deployment for live traffic video analysis.
