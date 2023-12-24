import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
df = pd.read_csv('training_data.csv')

# Định nghĩa mô hình
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# Tạo vectơ nhúng Word2Vec từ văn bản
sentences = [sentence.split() for sentence in df['question']]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    sentences, df['intent'].astype('category').cat.codes.values, test_size=0.2, random_state=42
)

# Chuyển đổi câu hỏi thành vectơ nhúng Word2Vec
def embed_sentence(sentence, word2vec_model):
    embedded_sentence = [word2vec_model.wv[word] for word in sentence if word in word2vec_model.wv]
    if len(embedded_sentence) == 0:
        # Return zero vector if no words are in the Word2Vec model
        return np.zeros(word2vec_model.vector_size)
    return np.mean(embedded_sentence, axis=0)

X_train_embedded = [embed_sentence(sentence, word2vec_model) for sentence in X_train]
X_test_embedded = [embed_sentence(sentence, word2vec_model) for sentence in X_test]

# Chuyển đổi dữ liệu thành tensor PyTorch
X_train = torch.tensor(X_train_embedded, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

# Tạo DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Khởi tạo mô hình
input_size = X_train.shape[1]
output_size = len(df['intent'].unique())
model = SimpleModel(input_size=input_size, output_size=output_size)

# Định nghĩa hàm mất mát và trình tối ưu hóa
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Huấn luyện mô hình
epochs = 10
losses = []
for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Xóa các gradients trước khi backpropagation
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass và cập nhật trọng số
        loss.backward()
        optimizer.step()

    # Lưu giá trị mất mát cho epoch hiện tại
    losses.append(loss.item())

    # In thông tin mất mát sau mỗi epoch
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Vẽ biểu đồ giảm mất mát qua các epoch
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Lưu trạng thái của mô hình
torch.save(model.state_dict(), 'trained_model.pth')

# Hàm dự đoán câu trả lời cho câu hỏi mới
def predict_answer(model, word2vec_model, question):
    # Chuyển câu hỏi mới thành vectơ nhúng Word2Vec
    new_question_embedded = embed_sentence(question.split(), word2vec_model)
    new_question_tensor = torch.tensor(new_question_embedded, dtype=torch.float32).unsqueeze(0)

    # Dự đoán intent cho câu hỏi mới
    with torch.no_grad():
        model.eval()
        output = model(new_question_tensor)
        predicted_intent = torch.argmax(output).item()

    # Lấy câu trả lời tương ứng với intent dự đoán
    answer = df[df['intent'].astype('category').cat.codes == predicted_intent]['answer'].values[0]

    return answer

# Dự đoán câu trả lời cho câu hỏi mới
new_question = input("Nhập câu hỏi mới: ")
predicted_answer = predict_answer(model, word2vec_model, new_question)

# In câu hỏi và câu trả lời dự đoán
print(f"Câu hỏi: {new_question}")
print(f"Câu trả lời dự đoán: {predicted_answer}")

# Dự đoán xác suất cho từng lớp
with torch.no_grad():
    model.eval()
    new_question_embedded = embed_sentence(new_question.split(), word2vec_model)
    new_question_tensor = torch.tensor(new_question_embedded, dtype=torch.float32).unsqueeze(0)
    output = model(new_question_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)

# In xác suất cho từng lớp
class_names = df['intent'].unique()
for i, prob in enumerate(probabilities.squeeze().tolist()):
    print(f"Xác suất cho lớp '{class_names[i]}': {prob:.4f}")
