```markdown
# Traffic Management using Federated Learning and YOLO Object Detection

This project simulates a federated learning setup for managing traffic lights, focusing on adaptive control based on real-time vehicle detection and classification. The YOLO model detects various vehicle types and adjusts traffic light durations according to vehicle type and count. Federated learning ensures data privacy by allowing decentralized model training.

## Installation

To get started, install the necessary libraries:

```bash
pip install torch torchvision torchaudio
pip install opencv-python-headless
pip install pyyaml
pip install tqdm
pip install matplotlib
pip install scikit-learn
pip install pandas
pip install seaborn
pip install ultralytics
```

## Project Overview

This project combines object detection, traffic analysis, and federated learning to create a smart traffic light system that dynamically adjusts green light duration based on vehicle density and type. The components of the project are:

1. **Data Collection and Preparation**
   - Upload a traffic video file.
   - Extract frames from the video to be used as input data.

2. **Object Detection with YOLO**
   - YOLO is used for real-time detection and classification of vehicles.
   - Detects multiple vehicle classes, including cars, motorcycles, buses, and trucks.

3. **Dataset Creation**
   - Process frames to create a labeled dataset of vehicle images.
   - Split the dataset into training and testing sets.

4. **Federated Learning with FedAvg Algorithm**
   - FedAvg is a federated learning algorithm that averages model parameters from multiple clients to create a global model.
   - Each client (simulated as a traffic light) trains a model locally and shares the updates with the global model.
   - A CNN model is used to classify vehicles detected by YOLO.

5. **Traffic Light Control System**
   - The `TrafficLightController` class determines green light duration based on vehicle type and count.
   - Assigns weights to heavier vehicles (bus, truck) for longer green durations.
   - Calculates green light duration based on weighted vehicle count, with a cap of 90 seconds.

6. **Evaluation**
   - Model accuracy is evaluated by comparing predictions to actual labels in the test set.

## Code Structure

### Vehicle Detection Function

The `detect_vehicles` function identifies vehicles in each frame and extracts vehicle images based on bounding boxes provided by the YOLO model. It returns detected vehicles with their respective classes, allowing for further classification and analysis.

```python
def detect_vehicles(frame):
    results = yolo_model(frame)
    vehicles = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                vehicle_img = frame[y1:y2, x1:x2]
                vehicles.append((vehicle_img, vehicle_classes[cls]))
    return vehicles
```

### Federated Learning with FedAvg Algorithm

The FedAvg algorithm aggregates model updates from multiple clients to update the global model. This allows for collaborative model training across different traffic lights while maintaining data privacy.

```python
def federated_avg(global_model, local_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([local_models[i].state_dict()[k].float() for i in range(len(local_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model
```

### Traffic Light Control System

The `TrafficLightController` class uses YOLO detection results to determine green light duration based on vehicle type and count. Heavier vehicles have higher weights, leading to extended green light times in proportion to traffic density.

```python
class TrafficLightController:
    def __init__(self, model):
        self.model = model
        self.vehicle_weights = {'car': 1, 'bus': 2, 'truck': 2}

    def process_frame(self, frame):
        vehicles = detect_vehicles(frame)
        vehicle_count = {'car': 0, 'bus': 0, 'truck': 0}

        for vehicle_img, vehicle_type in vehicles:
            vehicle_count[vehicle_type] += 1

        return vehicle_count

    def calculate_green_duration(self, vehicle_count):
        weighted_count = sum(count * self.vehicle_weights[v_type] for v_type, count in vehicle_count.items())
        base_duration = 30  # Base green light duration in seconds
        return min(base_duration + weighted_count * 2, 90)  # Cap at 90 seconds

    def control_traffic_light(self, frame):
        vehicle_count = self.process_frame(frame)
        green_duration = self.calculate_green_duration(vehicle_count)
        return green_duration
```

### Model Evaluation

To evaluate the global model, accuracy is calculated by comparing the predicted and actual labels in the test dataset.

```python
def evaluate_model(model, test_data):
    test_dataset = TrafficDataset(test_data, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
```

### Visualization

The `visualize_results` function displays detected vehicles along with the recommended green light duration.

```python
def visualize_results(frame, green_duration):
    vehicles = detect_vehicles(frame)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for vehicle_img, vehicle_type in vehicles:
        bbox = yolo_model(vehicle_img)[0].boxes[0].xyxy[0].tolist()
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(bbox[0], bbox[1], vehicle_type, color='r', fontweight='bold')

    ax.set_title(f"Traffic Analysis - Green Duration: {green_duration:.2f} seconds")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
```

## Conclusion

This project demonstrates the application of federated learning to improve traffic light management by utilizing YOLO-based vehicle detection. With a federated approach, the system can efficiently adapt to varying traffic conditions while preserving data privacy across different clients.

## Future Improvements

- **Implement more complex federated learning algorithms.**
- **Integrate more vehicle classes for precise traffic control.**
- **Incorporate real-time data streaming for continuous model updates.**
```
