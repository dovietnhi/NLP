import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd  # Thêm dòng này để import thư viện Pandas
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Định nghĩa mô hình
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# Hàm dự đoán câu trả lời cho câu hỏi mới
def predict_answer(model, tfidf_vectorizer, question):
    # Chuyển câu hỏi mới thành vectơ đặc trưng TF-IDF
    new_question_tfidf = tfidf_vectorizer.transform([question])
    new_question_tensor = torch.tensor(new_question_tfidf.toarray(), dtype=torch.float32)

    # Dự đoán intent cho câu hỏi mới
    with torch.no_grad():
        model.eval()
        output = model(new_question_tensor)
        predicted_intent = torch.argmax(output).item()

    # Lấy câu trả lời tương ứng với intent dự đoán
    answer = df[df['intent'].astype('category').cat.codes == predicted_intent]['answer'].values[0]

    return answer

# Đọc dữ liệu từ file CSV
df = pd.read_csv('training_data.csv')

# Tạo vectơ đặc trưng TF-IDF từ văn bản
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(df['question'])

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, df['intent'].astype('category').cat.codes.values, test_size=0.2, random_state=42
)

# Chuyển đổi dữ liệu thành tensor PyTorch
X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
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

# Dự đoán câu trả lời cho câu hỏi mới
new_question = input("Nhập câu hỏi mới: ")
predicted_answer = predict_answer(model, tfidf_vectorizer, new_question)

# In câu hỏi và câu trả lời dự đoán
print(f"Question: {new_question}")
print(f"Predicted Answer: {predicted_answer}")
