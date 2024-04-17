import torch

"""
В этом примере EarlyStoppingByAccuracy - это класс, который определяет условия 
остановки обучения на основе показателя accuracy. Метод __call__ вызывается на 
каждой эпохе с текущим значением accuracy, и если необходимые условия 
выполняются, обучение прерывается.
"""

class EarlyStoppingByAccuracy:
    def __init__(self, monitor='accuracy', value=0.99, patience=5, verbose=0):
        self.monitor = monitor
        self.value = value
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -float('inf') if 'acc' in monitor else float('inf')

    def __call__(self, epoch, accuracy):
        if 'acc' in self.monitor:
            if accuracy > self.best:
                self.best = accuracy
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    if self.verbose > 0:
                        print(f"Epoch {epoch}: early stopping")
                    return True
        else:
            if accuracy < self.best:
                self.best = accuracy
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    if self.verbose > 0:
                        print(f"Epoch {epoch}: early stopping")
                    return True
        return False
"""
# Использование:
# Замените monitor, value, patience на нужные вам значения
early_stopping = EarlyStoppingByAccuracy(monitor='accuracy', value=0.99, patience=5, verbose=1)

# Пример цикла обучения
for epoch in range(num_epochs):
    # Ваш код для обучения модели
    accuracy = ...  # Здесь должен быть ваш реальный показатель accuracy
    if early_stopping(epoch, accuracy):
        break  # Прекращаем обучение, если early stopping сработал
"""