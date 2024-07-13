
Group Number 4

Team Members: 

Roghieh Farajialamooti
Ghazaleh Hadian Ghahfarokhi
Tural Hasanov
Gökcer Sönmezocak

########## Please load the model with the code:

baseModel = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
baseModel.fc = nn.Sequential(nn.Linear(512, 64), nn.Dropout(p=0.5), nn.Linear(64, 1))
model_resnet18_classification = baseModel
model_resnet18_classification = model_resnet18_classification.to(device).eval()
model_resnet18_classification.load_state_dict(torch.load('model_resnet18_classification.pth'))

########## The input image of the model should be Tensor. In order to preprocess the image for the model run the following code:

image_path = '/content/Test.jpg'

class_names = ['class 0: Stop', 'class 1: Priority']

image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
image = image.unsqueeze(0)
image /= 255.0  

model_resnet18_classification.eval().to(device)
with torch.no_grad():
  outputs = model_resnet18_classification(image.to(device))
  predicted_prob = torch.sigmoid(outputs).item()  

if predicted_prob >= 0.5:
  class_name = class_names[1]  
else:
  class_name = class_names[0]  

plt.imshow(image.squeeze(0).permute(1, 2, 0).numpy())
plt.axis('off')
plt.show()

print('Predicted Class: ', class_name)

########## To see the metrics:

model_resnet18_classification.eval().to(device)

num_batches_test = (len(X_test) + batch_size - 1) // batch_size
actuals = []
predictions = []

with torch.no_grad():
  for i in range(num_batches_test):
    batch_inputs_test = X_test[i * batch_size:(i + 1) * batch_size]
    batch_labels_test = y_test[i * batch_size:(i + 1) * batch_size].unsqueeze(1).float().to(device)

    outputs = model_resnet18_classification(batch_inputs_test)
    probabilities = torch.sigmoid(outputs)
    predicted_labels = (probabilities > 0.5).float()
        
    actuals.extend(batch_labels_test.cpu().numpy())
    predictions.extend(predicted_labels.cpu().numpy())

actuals = np.array(actuals)
predictions = np.array(predictions)

accuracy = accuracy_score(actuals, predictions)
precision = precision_score(actuals, predictions, average='binary')
recall = recall_score(actuals, predictions, average='binary')
f1 = f1_score(actuals, predictions, average='binary')

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)