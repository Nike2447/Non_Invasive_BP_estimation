# AlexNet and ResNet for PPG Time-Series Regression

This repository contains implementations of **AlexNet** and **ResNet** models designed for regression tasks on **PPG (Photoplethysmogram) time-series data**. The goal is to predict Systolic Blood Pressure (SBP) and Diastolic Blood Pressure (DBP) values.

---

## **AlexNet-1D**
### **Model Description**
AlexNet-1D is a convolutional neural network adapted from the original AlexNet for 1D time-series data. The architecture includes:
- **Convolutional Layers**: 5 convolutional layers with ReLU activations and optional max-pooling.
- **Fully Connected Layers**: Two dense layers of size 4096 with dropout regularization.
- **Output Layers**:
  - One output for SBP prediction.
  - One output for DBP prediction.

### **Key Features**
- Handles time-series PPG data as input.
- Uses dropout (50%) to mitigate overfitting.
- Outputs SBP and DBP values for regression tasks.

---

## **ResNet-1D**
### **Model Description**
ResNet-1D is a 1D adaptation of the Residual Network architecture. It features residual connections to improve gradient flow and enable deeper networks. The architecture includes:
- **Residual Blocks**:
  - Two convolutional layers per block.
  - Shortcut connections to add the input tensor to the output of the block.
- **Global Average Pooling**: Reduces the feature map into a single vector per filter.
- **Output Layers**:
  - One dense layer for SBP prediction.
  - One dense layer for DBP prediction.

### **Key Features**
- Residual connections mitigate vanishing gradients and improve convergence.
- Scalable architecture with customizable filters and block sizes.
- Suitable for long time-series data.

---

## **Dataset**
- The models are trained on PPG time-series data stored in an HDF5 file format.
- **Structure**:
  - `ppg`: The PPG signal (time-series data).
  - `label`: Ground truth labels (SBP and DBP).
  - `subject_idx`: Subject IDs for grouping.

### **Loading the Dataset**
```python
import h5py

def load_dataset(filename, num_samples=None):
    with h5py.File(filename, 'r') as h5:
        ppg = h5['ppg'][:num_samples]
        labels = h5['label'][:num_samples]
        subject_idx = h5['subject_idx'][:num_samples]
    return ppg, labels, subject_idx
```

---

## **Training**
1. **Define Input Shape**:
   - For both models, the input shape is `(ppg_length, 1)`.

2. **Compile the Model**:
   - Loss: `Mean Absolute Error (MAE)` for SBP and DBP.
   - Optimizer: `Adam` optimizer.

3. **Train the Model**:
   ```python
   model.fit(
       x=train_data,
       y=[train_sbp, train_dbp],
       validation_data=(val_data, [val_sbp, val_dbp]),
       epochs=50,
       batch_size=32
   )
   ```

---

## **Evaluation Metrics**
- **Loss**:
  - `sbp_loss`: MAE for SBP predictions.
  - `dbp_loss`: MAE for DBP predictions.
  
- **Validation Metrics**:
  - `val_sbp_mae`: MAE for SBP on the validation set.
  - `val_dbp_mae`: MAE for DBP on the validation set.

---

## **Usage**
### **AlexNet-1D**
```python
model = AlexNet_1D(data_in_shape=(ppg_length, 1), num_output=2)
model.compile(optimizer='adam', loss='mae', metrics=['mae'])
```

### **ResNet-1D**
```python
model = ResNet1D(input_shape=(ppg_length, 1), num_blocks=[2, 2, 2], filters=[64, 128, 256])
model.compile(optimizer='adam', loss='mae', metrics=['mae'])
```

---

## **Key Benefits**
- **AlexNet-1D**: Simple, lightweight, and effective for shallow feature extraction.
- **ResNet-1D**: More advanced, deeper architecture with residual connections for improved accuracy and scalability.

---

## **References**
- [AlexNet Paper (2012)](https://doi.org/10.1145/3065386)
- [ResNet Paper (2015)](https://arxiv.org/abs/1512.03385)

