![[Pasted image 20260208135204.png]]

# code retrieval

https://arxiv.org/abs/2602.05550
собственный бенчмарк от ребят из итмо. Взяли модели энкодеров, использовали аугментацию кода метаданными, потом дообучили, используя контрастивное обучение (Multiple Negatives Ranking Loss)

https://arxiv.org/abs/2602.05892
бенчмарк для оценки агентов. В целом - есть много общего

Главное, с чем столкнулся - в проде используется не только поиск релевантного чанка, но и само разбиение на чанки. Для графового поиска такое необходимо. Без этого невозможно его использовать

плюс становятся невозможными такие идеи как:
1. проверка логичности и связности ответа
2. метрики эффективности. Насколько много лишнего нашла модель
3. использование в качестве верного не 1 чанка, а структуры
4. построение полной структуры кода, чтобы по ней искать, а не отдельных чанков


https://arxiv.org/abs/2601.11124
статья про дообучение энкодеров. Можно взять метрики и идеи

https://arxiv.org/abs/2512.05411
добавление метаданных с помощью llm

https://arxiv.org/abs/2602.03400
фильтрация метаданных

https://arxiv.org/abs/2409.14609
создание метаданных с помощью регулярок
# repository-level code retrieval

![[Pasted image 20260211141709.png]]

## Reliable Graph-RAG for Codebases: AST-Derived Graphs vs LLM-Extracted Knowledge Graphs

https://arxiv.org/abs/2601.08773
Использование AST для построения графа
Таким образом Граф всегда строится одинаково
## GraphCoder: Enhancing Repository-Level Code Completion via Code Context Graph-based Retrieval and Language Model

https://arxiv.org/abs/2406.07003
Пайплайн поиска - это поиск якоря, затем поиск соседей

## RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph

https://arxiv.org/abs/2410.14684
Идея межфайловых связей - граф строится в пределах всего репозитория. Функции разных файлов взаимосвязаны


## Context-Augmented Code Generation Using Programming Knowledge Graphs

https://arxiv.org/abs/2601.20810
Ограничение в виде радиуса поиска. Мы не идем дальше n связи. Таким образом сохраняем пямять

## Knowledge Graph Based Repository-Level Code Generation
https://arxiv.org/abs/2505.14394

# другие статьи

## Repository Intelligence Graph: Deterministic Architectural Map for LLM Code Assistants

https://arxiv.org/abs/2601.10112
