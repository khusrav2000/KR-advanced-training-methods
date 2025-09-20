# Advanced Training Algorithms for Deep Neural Networks

Реализация и исследование продвинутых алгоритмов обучения глубоких нейронных сетей в рамках курсовой работы.

## 📋 Обзор

Данный проект содержит реализацию пяти современных алгоритмов оптимизации нейронных сетей:

1. **FER (Reducing Flipping Errors)** - уменьшение ошибок переключения классификации
2. **Gradient Correction** - коррекция градиентов для улучшения сходимости  
3. **Forward Signal Propagation** - обучение через прямое распространение сигналов
4. **Selective Localized Learning** - селективное локализованное обучение
5. **Zero-Shot Hyperparameter Transfer** - перенос гиперпараметров без дообучения

Все алгоритмы реализованы в соответствии с API PyTorch для простой интеграции в существующие проекты.

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install torch torchvision numpy matplotlib seaborn pandas tqdm
```

### Базовое использование

```python
from advanced_optimizers import FEROptimizer
import torch
import torch.nn as nn

# Создание модели
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(), 
    nn.Linear(512, 10)
)

# Создание оптимизатора FER
optimizer = FEROptimizer(
    model.parameters(),
    base_optimizer=torch.optim.Adam,
    base_optimizer_kwargs={'lr': 0.001},
    fer_weight=0.1
)

# Обычный цикл обучения
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch.x), batch.y)
        loss.backward()
        optimizer.step()
```



## 🔬 Описание алгоритмов

### 1. FER (Reducing Flipping Errors)

Алгоритм направлен на уменьшение "переключений" классификации на правильно классифицированных образцах путем регуляризации градиентов.

**Особенности:**
- Память для хранения предыдущих градиентов
- Консистентность через косинусное сходство
- Минимальные вычислительные затраты

**Использование:**
```python
from advanced_optimizers import create_fer_optimizer

optimizer = create_fer_optimizer(
    model.parameters(),
    lr=0.001,
    fer_weight=0.1,
    consistency_threshold=0.9
)
```

### 2. Gradient Correction

Фреймворк коррекции градиентов через модули GC-W и GC-ODE для улучшения качества обновлений параметров.

**Особенности:**
- Модульная архитектура коррекции
- Обучаемые параметры коррекции
- Сокращение эпох обучения на ~20%

**Использование:**
```python
from advanced_optimizers import create_gradient_correction_optimizer

optimizer = create_gradient_correction_optimizer(
    model.parameters(),
    lr=0.001,
    use_gcw=True,
    use_gcode=True,
    device=device
)
```

### 3. Forward Signal Propagation

Альтернатива обратному распространению через прямое распространение обучающих сигналов.

**Особенности:**
- Биологически правдоподобное обучение
- Параллельное обучение слоев
- Специальные SigProp слои

**Использование:**
```python
from advanced_optimizers import SigPropNet, create_sigprop_optimizer

model = SigPropNet(input_dim=784, hidden_dims=[512, 256], output_dim=10)
optimizer = create_sigprop_optimizer(model, lr=0.001, signal_lr=0.0001)
```

### 4. Selective Localized Learning

Комбинирует локализованное (Хеббовское) обучение с SGD, селективно выбирая слои для каждого типа обновлений.

**Особенности:**
- Ускорение обучения в 1.5 раза
- Динамический выбор режима обучения
- Экономия памяти

**Использование:**
```python
from advanced_optimizers import create_localized_optimizer

optimizer = create_localized_optimizer(
    model,
    lr=0.001,
    hebbian_lr=0.0001,
    selection_mode='dynamic'
)
```

### 5. Zero-Shot Hyperparameter Transfer

μTransfer для переноса гиперпараметров с малых моделей на большие без дополнительной настройки.

**Особенности:**
- Maximal Update Parametrization (μP)
- Автоматическое масштабирование learning rate
- Радикальное сокращение затрат на настройку

**Использование:**
```python
from advanced_optimizers import create_mu_transfer_optimizer

optimizer = create_mu_transfer_optimizer(
    large_model,
    base_model_config={'width': 64, 'depth': 4},
    target_model_config={'width': 512, 'depth': 12},
    base_hyperparams={'lr': 0.001, 'weight_decay': 0.01}
)
```
*Этот проект является результатом исследования современных методов оптимизации нейронных сетей и демонстрирует практическую реализацию теоретических разработок в области машинного обучения.*
